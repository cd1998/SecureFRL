
from seal import *
import numpy as np
import math
from tqdm import tqdm
import time

NUM_CLIENTS = 25
POLY_MOD_DEGREE = 8192
PLAIN_MOD_BIT_SIZE = 35

# PM
parms_pm = EncryptionParameters(scheme_type.bfv)
parms_pm.set_poly_modulus_degree(POLY_MOD_DEGREE)
parms_pm.set_coeff_modulus(CoeffModulus.Create(POLY_MOD_DEGREE, [60]))
parms_pm.set_plain_modulus(PlainModulus.Batching(POLY_MOD_DEGREE, PLAIN_MOD_BIT_SIZE))
parms_pm.set_encoding_method(encoding_method.pm)
context_pm = SEALContext(parms_pm)

# PM
keygen_pm = KeyGenerator(context_pm)
sk_pm = keygen_pm.secret_key()
pk_pm = keygen_pm.create_public_key()
# relin_pm = keygen_pm.create_relin_keys()
enc_pm = Encryptor(context_pm, pk_pm)
eval_pm = Evaluator(context_pm)
dec_pm = Decryptor(context_pm, sk_pm)
batch_pm = BatchEncoder(context_pm)

slot_count = batch_pm.slot_count()

mnist_layer =[288,18432,1605632,1280]

# mnist_layer = [288,18432,1605632,1280]
# svhn_layer = [1728,36864,73728,147456,294912,589824,1179648,2359296,524288,65536,2560]
# cifar10_layer = [1728,36864,36864,36864,36864,73728,147456,8192,147456,147456,294912,589824,32768,589824,589824,1179648,2359296,131072,2359296,2359296,5120]


N = 3   # repeat N times

enc_times = []
robust_times = []
dec_times = []

for run in range(N):
    # data
    mnist_ranking_clients = []
    for cid in range(NUM_CLIENTS):
        client_ranking = []
        for size in mnist_layer:
            vec = np.arange(size)
            client_ranking.append(np.random.permutation(vec))
        mnist_ranking_clients.append(client_ranking)

    # ciphertext
    ciphertexts_pm_mnist = [[] for _ in range(NUM_CLIENTS)]

    num_ciphertexts = sum(
        math.ceil(len(num_list) / slot_count)
        for num_list in mnist_ranking_clients[0]
    )
    rotation_steps = np.random.randint(
        low=0,
        high=slot_count,
        size=num_ciphertexts
    )

    t1 = time.perf_counter()
    t_flip = 0.0
    for cid in range(NUM_CLIENTS):
        ct_idx = 0
        for num_list in mnist_ranking_clients[cid]:
            num_chunks = math.ceil(len(num_list) / slot_count)

            for i in range(num_chunks):
                start = i * slot_count
                end = min(start + slot_count, len(num_list))
                chunk = num_list[start:end]

                # padding
                if len(chunk) < slot_count:
                    chunk = np.pad(chunk, (0, slot_count - len(chunk)))

                v_i = rotation_steps[ct_idx]
                if v_i > 0:
                    # time1 = time.perf_counter()
                    chunk[-v_i:] = -chunk[-v_i:]
                    # time2 = time.perf_counter()
                    # t_flip += (time2 - time1)

                # encode & encrypt
                plain = batch_pm.encode(chunk)
                ciphertexts_pm_mnist[cid].append(enc_pm.encrypt(plain))
                ct_idx += 1
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    rotation_poly = []
    for i in rotation_steps:
        v = np.zeros(slot_count, dtype=np.int64)
        v[i] = 1
        rotation_poly.append(batch_pm.encode(v))

    # rotation
    rotated_ciphertexts = [[] for _ in range(NUM_CLIENTS)]
    for cid in range(NUM_CLIENTS):
        for k in range(num_ciphertexts):
            ct = ciphertexts_pm_mnist[cid][k]
            r = int(rotation_steps[k])
            if r != 0:
                ct_rot = eval_pm.multiply_plain(ct, rotation_poly[k])
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
                pt = dec_pm.decrypt(rotated_ciphertexts[cid][ct_idx])
                vec = batch_pm.decode(pt)
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
            pt_rand = batch_pm.encode(rand_vec)

            ct = ciphertexts_pm_mnist[cid][k]
            ct_add = eval_pm.add_plain(ct, pt_rand)

            masked_ciphertexts[cid].append(ct_add)
            random_masks[cid].append(pt_rand)

    # decrypt → re-encrypt
    refreshed_ciphertexts = {cid: [] for cid in valid_clients}

    for cid in valid_clients:
        for k in range(num_ciphertexts):
            pt = dec_pm.decrypt(masked_ciphertexts[cid][k])
            ct = enc_pm.encrypt(pt)
            refreshed_ciphertexts[cid].append(ct)

    # remove mask
    final_client_ciphertexts = {cid: [] for cid in valid_clients}

    for cid in valid_clients:
        for k in range(num_ciphertexts):
            ct = refreshed_ciphertexts[cid][k]
            ct_sub = eval_pm.sub_plain(ct, random_masks[cid][k])
            final_client_ciphertexts[cid].append(ct_sub)


    #element-wise add
    aggregated_pm_ciphertexts = []

    for k in range(num_ciphertexts):
        agg_ct = None
        for cid in valid_clients:
            ct = final_client_ciphertexts[cid][k]
            if agg_ct is None:
                agg_ct = ct
            else:
                eval_pm.add_inplace(agg_ct, ct)
        aggregated_pm_ciphertexts.append(agg_ct)

    print("PM complete robust aggregation finished.")
    t4 = time.perf_counter()

    t5 = time.perf_counter()
    aggregated_pm_plaintexts = {cid: [] for cid in valid_clients}
    for cid in valid_clients:
        for k in range(num_ciphertexts):
            pt = dec_pm.decrypt(aggregated_pm_ciphertexts[k])
            p = batch_pm.decode(pt)
            aggregated_pm_plaintexts[cid].append(p)
    t6 = time.perf_counter()
    
    enc_times.append(t2-t1)
    robust_times.append(t4-t3)
    dec_times.append(t6-t5)
    # print(aggregated_pm_plaintexts[0][0])
    # print(len(aggregated_pm_plaintexts[0][0]))
    # first_layer_vectors = [
    #     mnist_ranking_clients[cid][0] for cid in range(NUM_CLIENTS)
    # ]

    # # element-wise addition
    # first_layer_sum = np.sum(first_layer_vectors, axis=0)
    # print(first_layer_sum)
    # print(len(first_layer_sum))

 


print('encrypt+encode',np.mean(enc_times)/NUM_CLIENTS)
print('robust aggregation',np.mean(robust_times))
print('decrypt+decode',np.mean(dec_times)/NUM_CLIENTS)



