
from seal import *
import numpy as np
import math
from tqdm import tqdm
import time

NUM_CLIENTS = 25
POLY_MOD_DEGREE = 8192
PLAIN_MOD_BIT_SIZE = 35

# BFV
parms_bfv = EncryptionParameters(scheme_type.bfv)
parms_bfv.set_poly_modulus_degree(POLY_MOD_DEGREE)
parms_bfv.set_coeff_modulus(CoeffModulus.BFVDefault(POLY_MOD_DEGREE))
parms_bfv.set_plain_modulus(PlainModulus.Batching(POLY_MOD_DEGREE, PLAIN_MOD_BIT_SIZE))
context_bfv = SEALContext(parms_bfv)

keygen_bfv = KeyGenerator(context_bfv)
sk_bfv = keygen_bfv.secret_key()
pk_bfv = keygen_bfv.create_public_key()
rk_bfv = keygen_bfv.create_relin_keys()
gk_bfv = keygen_bfv.create_galois_keys()
enc_bfv = Encryptor(context_bfv, pk_bfv)
eval_bfv = Evaluator(context_bfv)
dec_bfv = Decryptor(context_bfv, sk_bfv)
batch_bfv = BatchEncoder(context_bfv)

slot_count = batch_bfv.slot_count()

mnist_layer = [1728,36864,36864,36864,36864,73728,147456,8192,147456,147456,294912,589824,32768,589824,589824,1179648,2359296,131072,2359296,2359296,5120]
# mnist_layer = [288,18432,1605632,1280]
# svhn_layer = [1728,36864,73728,147456,294912,589824,1179648,2359296,524288,65536,2560]
# cifar10_layer = [1728,36864,36864,36864,36864,73728,147456,8192,147456,147456,294912,589824,32768,589824,589824,1179648,2359296,131072,2359296,2359296,5120]

N = 1   # repeat N times

enc_times = []
robust_times = []
dec_times = []

for run in range(N):
    # data
    mnist_ranking_clients = []

    for cid in range(NUM_CLIENTS):
        client_ranking = []
        client_number = []
        for size in mnist_layer:
            vec = np.arange(size)
            client_ranking.append(np.random.permutation(vec))
        mnist_ranking_clients.append(client_ranking)

    # ciphertext
    ciphertexts_bfv_mnist = [[] for _ in range(NUM_CLIENTS)]

    num_ciphertexts = sum(
        math.ceil(len(num_list) / slot_count)
        for num_list in mnist_ranking_clients[0]
    )
    rotation_steps = np.random.randint(
        low=0,
        high=slot_count // 2,
        size=num_ciphertexts
    )

    t1 = time.perf_counter()
    for cid in range(NUM_CLIENTS):
        for num_list in mnist_ranking_clients[cid]:
            num_chunks = math.ceil(len(num_list) / slot_count)

            for i in range(num_chunks):
                start = i * slot_count
                end = min(start + slot_count, len(num_list))
                chunk = num_list[start:end]

                # padding
                if len(chunk) < slot_count:
                    chunk = np.pad(chunk, (0, slot_count - len(chunk)))

                # encode & encrypt
                plain = batch_bfv.encode(chunk)
                ciphertexts_bfv_mnist[cid].append(enc_bfv.encrypt(plain))
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    # rotation
    rotated_ciphertexts = [[] for _ in range(NUM_CLIENTS)]
    for cid in range(NUM_CLIENTS):
        for k in range(num_ciphertexts):
            ct = ciphertexts_bfv_mnist[cid][k]
            r = int(rotation_steps[k])
            if r != 0:
                ct_rot = eval_bfv.rotate_rows(ct, -rotation_steps[k], gk_bfv)
            rotated_ciphertexts[cid].append(ct_rot)

    # Decryption + Decoding + Concatenation + Verification
    valid_clients = []
    layer_sizes = mnist_layer
    for cid in range(NUM_CLIENTS):
        recovered_layers = []
        ct_idx = 0

        for layer_size in layer_sizes:
            needed = layer_size
            collected = []

            while needed > 0:
                pt = dec_bfv.decrypt(rotated_ciphertexts[cid][ct_idx])
                vec = batch_bfv.decode(pt)
                collected.extend(vec)
                needed -= len(vec)
                ct_idx += 1

            recovered_layers.append(collected)

        # Verification
        is_valid = True
        for layer_vec, layer_size in zip(recovered_layers, layer_sizes):
            vec = layer_vec
            non_zero = [v for v in vec if v != 0]

            if not all(1 <= v <= layer_size for v in non_zero):
                is_valid = False
                break

            if len(non_zero) != len(set(non_zero)):
                is_valid = False
                break

        if is_valid:
            valid_clients.append(cid)

    print(f"Valid clients after permutation check: {valid_clients}")

    # add mask
    masked_ciphertexts = {cid: [] for cid in valid_clients}
    random_masks = {cid: [] for cid in valid_clients}

    for cid in valid_clients:
        for k in range(num_ciphertexts):
            rand_vec = np.random.randint(
                low=0,
                high=2 ** 10,
                size=slot_count
            )
            pt_rand = batch_bfv.encode(rand_vec)

            ct = ciphertexts_bfv_mnist[cid][k]
            ct_add = eval_bfv.add_plain(ct, pt_rand)

            masked_ciphertexts[cid].append(ct_add)
            random_masks[cid].append(pt_rand)

    # decrypt → re-encrypt
    refreshed_ciphertexts = {cid: [] for cid in valid_clients}

    for cid in valid_clients:
        for k in range(num_ciphertexts):
            pt = dec_bfv.decrypt(masked_ciphertexts[cid][k])
            ct = enc_bfv.encrypt(pt)
            refreshed_ciphertexts[cid].append(ct)

    # remove mask
    final_client_ciphertexts = {cid: [] for cid in valid_clients}

    for cid in valid_clients:
        for k in range(num_ciphertexts):
            ct = refreshed_ciphertexts[cid][k]
            ct_sub = eval_bfv.sub_plain(ct, random_masks[cid][k])
            final_client_ciphertexts[cid].append(ct_sub)


    #element-wise add
    aggregated_bfv_ciphertexts = []

    for k in range(num_ciphertexts):
        agg_ct = None
        for cid in valid_clients:
            ct = final_client_ciphertexts[cid][k]
            if agg_ct is None:
                agg_ct = ct
            else:
                eval_bfv.add_inplace(agg_ct, ct)
        aggregated_bfv_ciphertexts.append(agg_ct)

    print("bfv complete robust aggregation finished.")
    t4 = time.perf_counter()

    t5 = time.perf_counter()
    aggregated_bfv_plaintexts = {cid: [] for cid in valid_clients}
    for cid in valid_clients:
        for k in range(num_ciphertexts):
            pt = dec_bfv.decrypt(aggregated_bfv_ciphertexts[k])
            p = batch_bfv.decode(pt)
            aggregated_bfv_plaintexts[cid].append(p)
    t6 = time.perf_counter()

    enc_times.append(t2-t1)
    robust_times.append(t4-t3)
    dec_times.append(t6-t5)


    # print(aggregated_bfv_plaintexts[0][0])
    # print(len(aggregated_bfv_plaintexts[0][0]))

    # first_layer_vectors = [
    #     mnist_ranking_clients[cid][0] for cid in range(NUM_CLIENTS)
    # ]

    # # element-wise addition
    # first_layer_sum = np.sum(first_layer_vectors, axis=0)
    # print(first_layer_sum)
    # print(len(first_layer_sum))


print('encrypt+encode',(t2-t1)/NUM_CLIENTS)
print('robust aggregation',t4-t3)
print('decrypt+decode',(t6-t5)/NUM_CLIENTS)






