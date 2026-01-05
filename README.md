# SecureFRL
# SecureFRL Core Code

This repository contains the **core implementation of the paper _SecureFRL_**.

The project integrates our **proposed novel encoding and decoding methods** into the **Microsoft SEAL homomorphic encryption library**, enabling efficient and secure computation for federated rank learning and related tasks.

---

## Overview

In this work, we extend the SEAL library by incorporating customized **encoding and decoding mechanisms** tailored to the requirements of SecureFRL. These modifications enable improved efficiency and flexibility when performing encrypted operations in privacy-preserving federated learning scenarios.

To bridge the gap between **C++-based SEAL** and **Python-based experimental workflows**, we leverage **SEAL-Python** as a binding interface. This design allows seamless invocation of SEAL functionalities directly from the Python environment, facilitating rapid prototyping and evaluation.

---

## Project Structure


