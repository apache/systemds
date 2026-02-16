# Shampoo Experiments

These experiments evaluate an implementation of the Shampoo optimizer, as described in: Gupta et al., “Shampoo: Preconditioned Stochastic Tensor Optimization” (<https://arxiv.org/abs/1802.09568>).
Shampoo is a second-order optimizer that preconditions gradients using estimates of the row- and column-wise gradient covariance. Compared to first-order optimizers such as SGD and Adam, Shampoo can converge faster, but it is typically more memory- and compute-intensive.

Experimental Setup:
    - Model: A simple two-layer CNN.
    - Datasets: MNIST and CIFAR-10.
    - CIFAR-10 regularization:
    To reduce overfitting during the 200 training epochs:
        - the CIFAR-10 training set is augmented (original + augmented copy),
        - a dropout layer is used.

Implemented Shampoo Variants_
    - Following Gupta et al., three Shampoo variants were implemented:
        - Classic Shampoo (baseline)
        - Shampoo + Momentum
        - Heuristic Shampoo: momentum + computing the expensive matrix-root steps only every n updates (to reduce compute cost)

    Additionally, the diagonal approximation (avoiding full eigendecomposition) is considered where applicable.

Baselines and Hyperparameter Tuning_
    - The Shampoo variants are compared against three common optimizers:
        - SGD
        - Adam
        - Adagrad
    - For every optimizer, hyperparameter tuning was performed. Importantly, no optimizer was tuned more extensively than the others. This means the results may not represent the absolute best achievable performance for each method, but the comparison better reflects:
        - how sensitive each optimizer is to tuning, and
        - how easy it is to obtain strong results in practice.

Results Overview

    A full overview of results (plots) is provided in: experiment_results.ipynb

    - Summary
        - MNIST
            - The Shampoo variants perform very well and typically converge faster.
            - A single update step is slightly slower, but fewer epochs may be required.
        - CIFAR-10
            - Shampoo variants perform worse than SGD and Adam in this setup.
            - A possible reason is that Shampoo benefits less in regimes where the model is underfitting or the setting strongly favors first-order methods.
        - Diagonal Shampoo
            - Slightly worse final performance than full Shampoo,
            - Reduces compute cost and improves time-to-convergence.
        - Heuristic Shampoo
            - Saves compute time, but performs worse in these experiments.
            - The diagonal variant often provides better trade-offs (faster and better-performing than the heuristic here).
            - Note: The original paper reports strong heuristic performance. The weaker results here may be due to the simplicity of the CNN or the specific dataset/training setup.

How to Run

    1. Create the CIFAR-10 CSV dataset (needed for CIFAR experiments):
        - Run: cifar_creation.py
    2. Single-step timing benchmark (optimizer update time):
        - Run: shampoo_experiment_time.dml
    3. Full training experiments (main benchmark):
        - Run: shamoo_optimizer_experiments.dml
        - At the end of the file, you can select:
            - standard experiments, or
            - hyperparameter fine-tuning runs (with relevant parameters)
        - For standard experiments, you can also adjust parameters in:
            - definingTrainingParameters
            - definingModelParameters
            - definingData
        - Adjust the values in the function "getOptimizer" to change the optimizers that are supposed to be used
        - Note: Default values are already tuned, changing them is not recommended unless you intentionally want a different experimental setup.
    4. (Optional) ResNet experiments
        - If you have enough compute, you can use: shampoo_optimizer_experiments_resnet.dml
        - Be aware: this pipeline is not optimized. It may be necessary to modify "definingData" to use more data and adjust dataset handling.
