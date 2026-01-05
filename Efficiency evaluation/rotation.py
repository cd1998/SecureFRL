from seal import *
import numpy as np
from seal import *
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil, os, gc
import statistics
from scipy.interpolate import make_interp_spline
import json
import csv

POLY_MOD_DEGREE = 4096
ROT_SAMPLE_TARGET = 4096
PLAIN_MOD_BIT_SIZE = 35
REP = 1  # repeat each micro-benchmark this many times and take median
OUTPUT_DIR = '.'
# ----------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ------------------ Rotation timing vs rotation amount ------------------
max_t_pm = slot_count -1
max_t_ori = slot_count // 2 -1
step_pm = max(1, max_t_pm // ROT_SAMPLE_TARGET)
step_ori = max(1, max_t_ori // ROT_SAMPLE_TARGET)

rot_t_values_pm = list(range(1, max_t_pm + 1, step_pm))
rot_t_values_ori = list(range(1, max_t_ori + 1, step_ori))

rot_times_pm = []
rot_times_bgv = []
rot_times_bfv = []
rot_times_ckks = []

for t in rot_t_values_pm:
    v = np.zeros(slot_count, dtype=np.int64)
    v[t] = 1
    v_plain = batch_pm.encode(v)
    time_pm_list = []
    for _ in range(REP):
        rand_data = np.random.randint(2**25, 2**26, size=slot_count, dtype=np.int64)
        plain_rand_pm = batch_pm.encode(rand_data)
        cipher_rand_pm = enc_pm.encrypt(plain_rand_pm)
        t0 = time.perf_counter()
        eval_pm.multiply_plain(cipher_rand_pm, v_plain)
        t1 = time.perf_counter()
        time_pm_list.append(t1-t0)
    mt_pm = statistics.mean(time_pm_list)
    rot_times_pm.append(mt_pm)

median_val = np.median(rot_times_pm)
threshold = median_val * 3

non_outliers = [v for v in rot_times_pm if v <= threshold]
    
if not non_outliers:  
    replacement_value = median_val
    non_outliers = [median_val]
else:
    replacement_value = np.mean(non_outliers)
    

cleaned_pm = [v if v <= threshold else replacement_value for v in rot_times_pm]

for t in rot_t_values_ori:
    time_bgv_list = []
    time_bfv_list = []
    time_ckks_list = []
    for _ in range(REP):
        rand_data = np.random.randint(2**25, 2**26, size=slot_count, dtype=np.int64)
        plain_rand_bgv = batch_bgv.encode(rand_data)
        cipher_rand_bgv = enc_bgv.encrypt(plain_rand_bgv)
        plain_rand_bfv = batch_bfv.encode(rand_data)
        cipher_rand_bfv = enc_bfv.encrypt(plain_rand_bfv)
        plain_rand_ckks = batch_ckks.encode(rand_data[:slot_count//2],scale)
        cipher_rand_ckks = enc_ckks.encrypt(plain_rand_ckks)

        t0 = time.perf_counter()
        eval_bgv.rotate_rows(cipher_rand_bgv, -t,gk_bgv)
        t1 = time.perf_counter()
        time_bgv_list.append(t1-t0)

        t0 = time.perf_counter()
        eval_bfv.rotate_rows(cipher_rand_bfv, -t,gk_bfv)
        t1 = time.perf_counter()
        time_bfv_list.append(t1-t0)

        t0 = time.perf_counter()
        eval_ckks.rotate_vector(cipher_rand_ckks, -t,gk_ckks)
        t1 = time.perf_counter()
        time_ckks_list.append(t1-t0)

    mt_bgv = statistics.mean(time_bgv_list)
    rot_times_bgv.append(mt_bgv)
    mt_bfv = statistics.mean(time_bfv_list)
    rot_times_bfv.append(mt_bfv)
    mt_ckks = statistics.mean(time_ckks_list)
    rot_times_ckks.append(mt_ckks)


median_val_bgv = np.median(rot_times_bgv)
threshold_bgv = median_val_bgv * 3  
median_val_bfv = np.median(rot_times_bfv)
threshold_bfv = median_val_bfv * 3  
median_val_ckks = np.median(rot_times_ckks)
threshold_ckks = median_val_ckks * 3 

non_outliers_bgv = [v for v in rot_times_bgv if v <= threshold_bgv]
non_outliers_bfv = [v for v in rot_times_bfv if v <= threshold_bfv]
non_outliers_ckks = [v for v in rot_times_ckks if v <= threshold_ckks]
    
replacement_value_bgv = np.mean(non_outliers_bgv)
replacement_value_bfv = np.mean(non_outliers_bfv)
replacement_value_ckks = np.mean(non_outliers_ckks)
    

cleaned_bgv = [v if v <= threshold_bgv else replacement_value_bgv for v in rot_times_bgv]
cleaned_bfv = [v if v <= threshold_bfv else replacement_value_bfv for v in rot_times_bfv]
cleaned_ckks = [v if v <= threshold_ckks else replacement_value_ckks for v in rot_times_ckks]
    
plt.figure(figsize=(12, 6))

def smooth_curve(x, y, points=300):
    x = np.array(x)
    y = np.array(y)
    X_ = np.linspace(x.min(), x.max(), points)
    Y_ = make_interp_spline(x, y)(X_)
    return X_, Y_
LW = 2.0
# RLWE-AHE
x_s, y_s = smooth_curve(rot_t_values_pm, [v*1000 for v in cleaned_pm])
plt.plot(x_s, y_s, linewidth=2.5,label='RLWE-AHE')

# BGV
x_s, y_s = smooth_curve(rot_t_values_ori, [v*1000 for v in cleaned_bgv])
plt.plot(x_s, y_s, linewidth=LW,label='BGV')

# BFV
x_s, y_s = smooth_curve(rot_t_values_ori, [v*1000 for v in cleaned_bfv])
plt.plot(x_s, y_s, linewidth=LW,label='BFV')

# CKKS
x_s, y_s = smooth_curve(rot_t_values_ori, [v*1000 for v in cleaned_ckks])
plt.plot(x_s, y_s, linewidth=LW,label='CKKS')

plt.xlabel('Rotation amount')
plt.ylabel('Time (ms)')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

fn1 = os.path.join(OUTPUT_DIR, f'rotation_time_{POLY_MOD_DEGREE}.png')
fn2 = os.path.join(OUTPUT_DIR, f'rotation_time_{POLY_MOD_DEGREE}.pdf')
plt.savefig(fn1, dpi=300)
plt.savefig(fn2)
print(f"Saved: {fn2}")

print('pm',statistics.mean(cleaned_pm),max(cleaned_pm),min(cleaned_pm))
print('bgv',statistics.mean(cleaned_bgv),max(cleaned_bgv),min(cleaned_bgv))
print('bfv',statistics.mean(cleaned_bfv),max(cleaned_bfv),min(cleaned_bfv))
print('ckks',statistics.mean(cleaned_ckks),max(cleaned_ckks),min(cleaned_ckks))

indices = list(range(1, len(cleaned_pm) + 1))

with open(f"cleaned_results_{POLY_MOD_DEGREE}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(indices)
    writer.writerow(cleaned_pm)
    writer.writerow(cleaned_bgv)
    writer.writerow(cleaned_bfv)
    writer.writerow(cleaned_ckks)

gc.enable()

