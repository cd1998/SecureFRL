from seal import *
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil, os, gc
import statistics

# ------------------------- User-tunable parameters -------------------------
POLY_MOD_DEGREE = 4096
PLAIN_MOD_BIT_SIZE = 35
REP = 1000

p = psutil.Process()
p.cpu_affinity([64])
gc.disable()

# ---------- Create four parameter sets: RLWE-AHE, BGV, BFV, CKKS ----------
parms_pm = EncryptionParameters(scheme_type.bfv)
parms_pm.set_poly_modulus_degree(POLY_MOD_DEGREE)
#parms_pm.set_coeff_modulus(CoeffModulus.BFVDefault(POLY_MOD_DEGREE))
parms_pm.set_coeff_modulus(CoeffModulus.Create(POLY_MOD_DEGREE, [60]))
parms_pm.set_plain_modulus(PlainModulus.Batching(POLY_MOD_DEGREE, PLAIN_MOD_BIT_SIZE))
parms_pm.set_encoding_method(encoding_method.pm)
context_pm = SEALContext(parms_pm)
# -------------------------------------------

parms_bgv = EncryptionParameters(scheme_type.bgv)
parms_bgv.set_poly_modulus_degree(POLY_MOD_DEGREE)
parms_bgv.set_coeff_modulus(CoeffModulus.BFVDefault(POLY_MOD_DEGREE))
#parms_bgv.set_coeff_modulus(CoeffModulus.Create(POLY_MOD_DEGREE, [60,20]))
parms_bgv.set_plain_modulus(PlainModulus.Batching(POLY_MOD_DEGREE, PLAIN_MOD_BIT_SIZE))
# do NOT call set_encoding_method here -> use default/original encoding
context_bgv = SEALContext(parms_bgv)
# -------------------------------------------

parms_bfv = EncryptionParameters(scheme_type.bfv)
parms_bfv.set_poly_modulus_degree(POLY_MOD_DEGREE)
parms_bfv.set_coeff_modulus(CoeffModulus.BFVDefault(POLY_MOD_DEGREE))
#parms_bfv.set_coeff_modulus(CoeffModulus.Create(POLY_MOD_DEGREE, [60]))
parms_bfv.set_plain_modulus(PlainModulus.Batching(POLY_MOD_DEGREE, PLAIN_MOD_BIT_SIZE))
context_bfv = SEALContext(parms_bfv)

# ----------------------------------------------------

parms_ckks = EncryptionParameters(scheme_type.ckks)
parms_ckks.set_poly_modulus_degree(POLY_MOD_DEGREE)
#parms_ckks.set_coeff_modulus(CoeffModulus.Create(POLY_MOD_DEGREE, [60, 40, 40, 60]))
parms_ckks.set_coeff_modulus(CoeffModulus.BFVDefault(POLY_MOD_DEGREE))
scale = 2.0 ** 40
context_ckks = SEALContext(parms_ckks)

# Keygens / helpers for both
keygen_pm = KeyGenerator(context_pm)
sk_pm = keygen_pm.secret_key()
pk_pm = keygen_pm.create_public_key()
# relin_pm = keygen_pm.create_relin_keys()
enc_pm = Encryptor(context_pm, pk_pm)
eval_pm = Evaluator(context_pm)
dec_pm = Decryptor(context_pm, sk_pm)
batch_pm = BatchEncoder(context_pm)

keygen_bgv = KeyGenerator(context_bgv)
sk_bgv = keygen_bgv.secret_key()
pk_bgv = keygen_bgv.create_public_key()
rk_bgv = keygen_bgv.create_relin_keys()
gk_bgv = keygen_bgv.create_galois_keys()
enc_bgv = Encryptor(context_bgv, pk_bgv)
eval_bgv = Evaluator(context_bgv)
dec_bgv = Decryptor(context_bgv, sk_bgv)
batch_bgv = BatchEncoder(context_bgv)

keygen_bfv = KeyGenerator(context_bfv)
sk_bfv = keygen_bfv.secret_key()
pk_bfv = keygen_bfv.create_public_key()
rk_bfv = keygen_bfv.create_relin_keys()
gk_bfv = keygen_bfv.create_galois_keys()
enc_bfv = Encryptor(context_bfv, pk_bfv)
eval_bfv = Evaluator(context_bfv)
dec_bfv = Decryptor(context_bfv, sk_bfv)
batch_bfv = BatchEncoder(context_bfv)

keygen_ckks = KeyGenerator(context_ckks)
sk_ckks = keygen_ckks.secret_key()
pk_ckks = keygen_ckks.create_public_key()
rk_ckks = keygen_ckks.create_relin_keys()
gk_ckks = keygen_ckks.create_galois_keys()
enc_ckks = Encryptor(context_ckks, pk_ckks)
eval_ckks = Evaluator(context_ckks)
dec_ckks = Decryptor(context_ckks, sk_ckks)
batch_ckks = CKKSEncoder(context_ckks)

slot_count = batch_pm.slot_count()
#------------------------------------------------------------
def remove_outliers_and_mean(values):
    if not values:
        return None, 0
    threshold = 3
    median_val = statistics.median(values)
    outliers = [v for v in values if v > median_val * threshold]
    filtered = [v for v in values if v <= median_val * threshold]

    if len(filtered) == 0:
        mean_val = statistics.mean(values)
    else:
        mean_val = statistics.mean(filtered)

    return mean_val, len(outliers)
#-----------------------------------------------------------------
encode_time_pm = []
encrypt_time_pm = []
addition_time_pm = []
decrypt_time_pm = []
decode_time_pm = []

encode_time_bgv = []
encrypt_time_bgv = []
addition_time_bgv = []
decrypt_time_bgv = []
decode_time_bgv = []

encode_time_bfv = []
encrypt_time_bfv = []
addition_time_bfv = []
decrypt_time_bfv = []
decode_time_bfv = []

encode_time_ckks = []
encrypt_time_ckks = []
addition_time_ckks = []
decrypt_time_ckks = []
decode_time_ckks = []

for _ in range(REP):
    rand_data = np.random.randint(2 ** 25, 2 ** 26, size=slot_count, dtype=np.int64)
    # print('rand_data',rand_data[:10])
    
    # PM scheme
    t0 = time.perf_counter()
    plain_rand_pm = batch_pm.encode(rand_data)
    t1 = time.perf_counter()
    encode_time_pm.append(t1-t0)
    
    t2 = time.perf_counter()
    cipher_rand_pm = enc_pm.encrypt(plain_rand_pm)
    t3 = time.perf_counter()
    encrypt_time_pm.append(t3-t2)

    
    raw_size = cipher_rand_pm.save_size()  
    print('raw_size',raw_size)


    # BGV scheme
    t0 = time.perf_counter()
    plain_rand_bgv = batch_bgv.encode(rand_data)
    t1 = time.perf_counter()
    encode_time_bgv.append(t1-t0)
    
    t2 = time.perf_counter()
    cipher_rand_bgv = enc_bgv.encrypt(plain_rand_bgv)
    t3 = time.perf_counter()
    encrypt_time_bgv.append(t3-t2)

    raw_size = cipher_rand_bgv.save_size()  
    print('raw_size',raw_size)


    # BFV scheme
    t0 = time.perf_counter()
    plain_rand_bfv = batch_bfv.encode(rand_data)
    t1 = time.perf_counter()
    encode_time_bfv.append(t1-t0)
    
    t2 = time.perf_counter()
    cipher_rand_bfv = enc_bfv.encrypt(plain_rand_bfv)
    t3 = time.perf_counter()
    encrypt_time_bfv.append(t3-t2)

    raw_size = cipher_rand_bfv.save_size()  
    print('raw_size',raw_size)

    # CKKS scheme
    t0 = time.perf_counter()
    plain_rand_ckks = batch_ckks.encode(rand_data[:slot_count//2], scale)
    t1 = time.perf_counter()
    encode_time_ckks.append(t1-t0)
    
    t2 = time.perf_counter()
    cipher_rand_ckks = enc_ckks.encrypt(plain_rand_ckks)
    t3 = time.perf_counter()
    encrypt_time_ckks.append(t3-t2)

    raw_size = cipher_rand_ckks.save_size() 
    print('raw_size',raw_size)

    #=================================================
    t0 = time.perf_counter()
    cipher_add_pm = eval_pm.add(cipher_rand_pm, cipher_rand_pm)
    t1 = time.perf_counter()
    addition_time_pm.append(t1-t0)

    t0 = time.perf_counter()
    cipher_add_bgv = eval_bgv.add(cipher_rand_bgv, cipher_rand_bgv)
    t1 = time.perf_counter()
    addition_time_bgv.append(t1-t0)

    t0 = time.perf_counter()
    cipher_add_bfv = eval_bfv.add(cipher_rand_bfv, cipher_rand_bfv)
    t1 = time.perf_counter()
    addition_time_bfv.append(t1-t0)

    t0 = time.perf_counter()
    cipher_add_ckks = eval_ckks.add(cipher_rand_ckks, cipher_rand_ckks)
    t1 = time.perf_counter()
    addition_time_ckks.append(t1-t0)
    #===================================================

    t0 = time.perf_counter()
    decrypt_rand_pm = dec_pm.decrypt(cipher_rand_pm)
    t1 = time.perf_counter()
    decrypt_time_pm.append(t1-t0)
    
    t2 = time.perf_counter()
    decode_rand_pm = batch_pm.decode(decrypt_rand_pm)
    t3 = time.perf_counter()
    decode_time_pm.append(t3-t2)
    # print('rand_data', decode_rand_pm[:10])

    # BGV scheme
    t0 = time.perf_counter()
    decrypt_rand_bgv = dec_bgv.decrypt(cipher_rand_bgv)
    t1 = time.perf_counter()
    decrypt_time_bgv.append(t1-t0)
    
    t2 = time.perf_counter()
    decode_rand_bgv = batch_bgv.decode(decrypt_rand_bgv)
    t3 = time.perf_counter()
    decode_time_bgv.append(t3-t2)
    # print('rand_data', decode_rand_bgv[:10])

    # BFV scheme
    t0 = time.perf_counter()
    decrypt_rand_bfv = dec_bfv.decrypt(cipher_rand_bfv)
    t1 = time.perf_counter()
    decrypt_time_bfv.append(t1-t0)
    
    t2 = time.perf_counter()
    decode_rand_bfv = batch_bfv.decode(decrypt_rand_bfv)
    t3 = time.perf_counter()
    decode_time_bfv.append(t3-t2)
    # print('rand_data', decode_rand_bfv[:10])

    # CKKS scheme
    t0 = time.perf_counter()
    decrypt_rand_ckks = dec_ckks.decrypt(cipher_rand_ckks)
    t1 = time.perf_counter()
    decrypt_time_ckks.append(t1-t0)
    
    t2 = time.perf_counter()
    decode_rand_ckks = batch_ckks.decode(decrypt_rand_ckks)
    t3 = time.perf_counter()
    decode_time_ckks.append(t3-t2)
    # print('rand_data', decode_rand_ckks[:10])


encode_mean_time_pm, n_encode_pm = remove_outliers_and_mean(encode_time_pm)
encrypt_mean_time_pm, n_encrypt_pm = remove_outliers_and_mean(encrypt_time_pm)
addition_mean_time_pm, n_add_pm = remove_outliers_and_mean(addition_time_pm)
decrypt_mean_time_pm, n_decrypt_pm = remove_outliers_and_mean(decrypt_time_pm)
decode_mean_time_pm, n_decode_pm = remove_outliers_and_mean(decode_time_pm)

encode_mean_time_bgv, n_encode_bgv = remove_outliers_and_mean(encode_time_bgv)
encrypt_mean_time_bgv, n_encrypt_bgv = remove_outliers_and_mean(encrypt_time_bgv)
addition_mean_time_bgv, n_add_bgv = remove_outliers_and_mean(addition_time_bgv)
decrypt_mean_time_bgv, n_decrypt_bgv = remove_outliers_and_mean(decrypt_time_bgv)
decode_mean_time_bgv, n_decode_bgv = remove_outliers_and_mean(decode_time_bgv)

encode_mean_time_bfv, n_encode_bfv = remove_outliers_and_mean(encode_time_bfv)
encrypt_mean_time_bfv, n_encrypt_bfv = remove_outliers_and_mean(encrypt_time_bfv)
addition_mean_time_bfv, n_add_bfv = remove_outliers_and_mean(addition_time_bfv)
decrypt_mean_time_bfv, n_decrypt_bfv = remove_outliers_and_mean(decrypt_time_bfv)
decode_mean_time_bfv, n_decode_bfv = remove_outliers_and_mean(decode_time_bfv)

encode_mean_time_ckks, n_encode_ckks = remove_outliers_and_mean(encode_time_ckks)
encrypt_mean_time_ckks, n_encrypt_ckks = remove_outliers_and_mean(encrypt_time_ckks)
addition_mean_time_ckks, n_add_ckks = remove_outliers_and_mean(addition_time_ckks)
decrypt_mean_time_ckks, n_decrypt_ckks = remove_outliers_and_mean(decrypt_time_ckks)
decode_mean_time_ckks, n_decode_ckks = remove_outliers_and_mean(decode_time_ckks)

print("\n=== encoding time ===")
print(f"PM:  {encode_mean_time_pm:.6f}s, outliers number: {n_encode_pm}")
print(f"BGV: {encode_mean_time_bgv:.6f}s, outliers number: {n_encode_bgv}")
print(f"BFV: {encode_mean_time_bfv:.6f}s, outliers number: {n_encode_bfv}")
print(f"CKKS:{encode_mean_time_ckks:.6f}s, outliers number: {n_encode_ckks}")

print("\n=== encryption time ===")
print(f"PM:  {encrypt_mean_time_pm:.6f}s, outliers number: {n_encrypt_pm}")
print(f"BGV: {encrypt_mean_time_bgv:.6f}s, outliers number: {n_encrypt_bgv}")
print(f"BFV: {encrypt_mean_time_bfv:.6f}s, outliers number: {n_encrypt_bfv}")
print(f"CKKS:{encrypt_mean_time_ckks:.6f}s, 异outliers number: {n_encrypt_ckks}")

print("\n=== addition time ===")
print(f"PM:  {addition_mean_time_pm:.6f}s, outliers number: {n_add_pm}")
print(f"BGV: {addition_mean_time_bgv:.6f}s, outliers number: {n_add_bgv}")
print(f"BFV: {addition_mean_time_bfv:.6f}s, 异outliers number: {n_add_bfv}")
print(f"CKKS:{addition_mean_time_ckks:.6f}s, outliers number: {n_add_ckks}")

print("\n=== decryption time ===")
print(f"PM:  {decrypt_mean_time_pm:.6f}s, outliers number: {n_decrypt_pm}")
print(f"BGV: {decrypt_mean_time_bgv:.6f}s, outliers number: {n_decrypt_bgv}")
print(f"BFV: {decrypt_mean_time_bfv:.6f}s, outliers number: {n_decrypt_bfv}")
print(f"CKKS:{decrypt_mean_time_ckks:.6f}s, outliers number: {n_decrypt_ckks}")

print("\n=== decoding time ===")
print(f"PM:  {decode_mean_time_pm:.6f}s, outliers number: {n_decode_pm}")
print(f"BGV: {decode_mean_time_bgv:.6f}s, outliers number: {n_decode_bgv}")
print(f"BFV: {decode_mean_time_bfv:.6f}s, outliers number: {n_decode_bfv}")
print(f"CKKS:{decode_mean_time_ckks:.6f}s, outliers number: {n_decode_ckks}")

total_encode_pm = encode_mean_time_pm + encrypt_mean_time_pm
total_decode_pm = decrypt_mean_time_pm + decode_mean_time_pm

total_encode_bgv = encode_mean_time_bgv + encrypt_mean_time_bgv
total_decode_bgv = decrypt_mean_time_bgv + decode_mean_time_bgv

total_encode_bfv = encode_mean_time_bfv + encrypt_mean_time_bfv
total_decode_bfv = decrypt_mean_time_bfv + decode_mean_time_bfv

total_encode_ckks = encode_mean_time_ckks + encrypt_mean_time_ckks
total_decode_ckks = decrypt_mean_time_ckks + decode_mean_time_ckks

print("\n=== encoding + encryption ===")
print(f"PM:  {total_encode_pm:.6f}s")
print(f"BGV: {total_encode_bgv:.6f}s")
print(f"BFV: {total_encode_bfv:.6f}s")
print(f"CKKS:{total_encode_ckks:.6f}s")

print("\n=== decryption + decoding ===")
print(f"PM:  {total_decode_pm:.6f}s")
print(f"BGV: {total_decode_bgv:.6f}s")
print(f"BFV: {total_decode_bfv:.6f}s")
print(f"CKKS:{total_decode_ckks:.6f}s")