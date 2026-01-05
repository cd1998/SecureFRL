import numpy as np
from tqdm import tqdm
from heu import phe
import time

# Paillier init
kit = phe.setup(phe.SchemaType.ZPaillier, 2048)
enc = kit.encryptor()
dec = kit.decryptor()
eval = kit.evaluator()
encoder = phe.FloatEncoder(phe.SchemaType.ZPaillier,scale=100)

N = 1000

xs = np.random.uniform(-1, 1, N)
ys = np.random.uniform(-1, 1, N)
scalars = np.random.uniform(-1, 1, N)

# =============================
# Encode
# =============================
t0 = time.perf_counter()
pts_x = []
for i in range(N):
    pts_x.append(encoder.encode(xs[i]))
t1 = time.perf_counter()
encode_time = (t1 - t0) / N

# =============================
# Encrypt
# =============================
t0 = time.perf_counter()
cts_x = []
for i in range(N):
    cts_x.append(enc.encrypt(pts_x[i]))
t1 = time.perf_counter()
encrypt_time = (t1 - t0) / N

# =============================
# Add (ct + ct)
# =============================
pts_y = [encoder.encode(ys[i]) for i in range(N)]
cts_y = [enc.encrypt(pts_y[i]) for i in range(N)]

t0 = time.perf_counter()
cts_add = []
for i in range(N):
    cts_add.append(eval.add(cts_x[i], cts_y[i]))
t1 = time.perf_counter()
add_time = (t1 - t0) / N

# =============================
# Mul (ct * plaintext scalar)
# =============================
pts_scalar = [encoder.encode(scalars[i]) for i in range(N)]

t0 = time.perf_counter()
cts_mul = []
for i in range(N):
    cts_mul.append(eval.mul(cts_x[i], pts_scalar[i]))
t1 = time.perf_counter()
mul_time = (t1 - t0) / N

# =============================
# Decrypt
# =============================
t0 = time.perf_counter()
pts_dec = []
for i in range(N):
    pts_dec.append(dec.decrypt(cts_x[i]))
t1 = time.perf_counter()
decrypt_time = (t1 - t0) / N

# =============================
# Decode
# =============================
t0 = time.perf_counter()
vals = []
for i in range(N):
    vals.append(encoder.decode(pts_dec[i]))
t1 = time.perf_counter()
decode_time = (t1 - t0) / N

# =============================
# Results
# =============================
print("==== Paillier (ZPaillier, 2048) Random Input Benchmark ====")
print(f"Encode   : {encode_time * 1e3:.3f} ms")
print(f"Encrypt  : {encrypt_time * 1e3:.3f} ms")
print(f"Add      : {add_time * 1e3:.3f} ms")
print(f"Mul      : {mul_time * 1e3:.3f} ms")
print(f"Decrypt  : {decrypt_time * 1e3:.3f} ms")
print(f"Decode   : {decode_time * 1e3:.3f} ms")
