# Extended Quickstart Guide

Welcome to the extended quickstart guide for Apache SystemDS. This quickstart page provides a high-level overview of both installation and points you to the detailed documentation for each path.

SystemDS can be installed and used in two different ways:

1. Using a **downloaded release**  
2. Using a **source build**

If you are primarily a user of SystemDS, start with the Release installation. If you plan to contribute or modify internals, follow the Source installation.

Each method is demonstrated in:
- Local mode  
- Spark mode  
- Federated mode (simple example)

For detailed configuration topics (BLAS, GPU, federated setup, contributing), see the links at the end.

---

# 1. Install from a Release

If you simply want to *use* SystemDS without modifying the source code, the recommended approach is to install SystemDS from an official Apache release.

**Full Release Installation Guide:** [SystemDS Install from release](https://apache.github.io/systemds/site/release_install.html)

# 2. Install from Source

If you plan to contribute to SystemDS or need to modify its internals, you can build SystemDS from source.

**Full Source Build Guide:** [SystemDS Install from source](https://apache.github.io/systemds/site/source_install.html)

# 3. After Installation

Once either installation path is completed, you can start running scripts:

- Local Mode - Run SystemDS locally
- Spark Mode - Execute scripts on Spark through `spark-submit`
- Federated Mode - Run operations on remote data using federated workers

For detailed commands and examples: [Execute SystemDS](https://apache.github.io/systemds/site/run_extended.html)

# 4. More Configuration

SystemDS provides advanced configuration options for performance tuning and specialized execution environments. 

- GPU Support — [GPU Guide](https://apache.github.io/systemds/site/gpu)  
- BLAS / Native Acceleration — [Native Backend (BLAS) Guide](https://apache.github.io/systemds/site/native-backend)  
- Federated Backend Deployment — [Federated Guide](https://apache.github.io/systemds/site/federated-monitoring.html)  
- Contributing to SystemDS — [Contributing Guide](https://github.com/apache/systemds/blob/main/CONTRIBUTING.md)

