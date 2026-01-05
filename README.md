# SecureFRL
## Overview
This repository contains the **core implementation of the paper SecureFRL: Efficient Privacy-Preserving Byzantine-Robust Federated Learning via Joint Training and Encryption Adaptation**.

In this work, we extend the SEAL library by incorporating customized **encoding and decoding mechanisms** tailored to the requirements of SecureFRL. These modifications enable improved efficiency and flexibility when performing encrypted operations in privacy-preserving federated learning scenarios.

To bridge the gap between **C++-based SEAL** and **Python-based experimental workflows**, we leverage **SEAL-Python** as a binding interface. This design allows seamless invocation of SEAL functionalities directly from the Python environment, facilitating rapid prototyping and evaluation.

All experimental and benchmarking code used for **efficiency evaluation** is located in the directory: Efficiency evaluation/
