# ResNet50 with LARS Optimizer

This document provides an overview of the ResNet50 implementation with the LARS (Layer-wise Adaptive Rate Scaling) optimizer in SystemDS.

## Overview

This script implements the ResNet50 architecture, a 50-layer deep convolutional neural network, and integrates it with the LARS optimizer for efficient large-batch training. ResNet architectures are known for their use of residual connections (shortcuts) to enable the training of very deep networks without suffering from vanishing gradients.

When combined with the LARS optimizer, this implementation is well-suited for large-scale image classification tasks, such as training on the ImageNet dataset.

## Key Features

- **ResNet50 Architecture**: A 50-layer deep CNN with residual connections.
- **LARS Optimizer**: Enables stable and efficient training with large batch sizes.
- **Bottleneck Design**: The building blocks of ResNet50 use a bottleneck design for improved efficiency.
- **Batch Normalization**: Used throughout the network to stabilize training.
- **Learning Rate Scheduling**: Can be combined with learning rate schedulers, such as one with warmup and polynomial decay, for optimal convergence.

## How to Use

To use the ResNet50-LARS implementation, you can source the script and call the training function with your data and desired hyperparameters.

```dml
source("nn/networks/resnet50_LARS.dml") as resnet50

# Load your data (e.g., X_train, Y_train)
# ...

# Initialize the model
model = resnet50::init(C=3, num_classes=1000, seed=42)

# Initialize the LARS optimizer state
optim_state = resnet50::init_lars_optim_params(model)

# Define hyperparameters
epochs = 100
batch_size = 4096
base_lr = 0.02 
trust_coeff = 0.001
# ... other hyperparameters ...

# Run the training loop
# ...
```

## Parameters

The main training function likely accepts the following parameters:

- `X_train`, `Y_train`: Training data and labels.
- `X_val`, `Y_val`: Validation data and labels.
- `epochs`: The number of training epochs.
- `batch_size`: The size of each training batch.
- `base_lr`: The base learning rate for the LARS optimizer.
- `trust_coeff`: The trust coefficient for the LARS optimizer.
- `weight_decay`: The L2 regularization strength.

*Note: This is a template README. Please update it with the specific details of the `resnet50_LARS.dml` implementation.* 