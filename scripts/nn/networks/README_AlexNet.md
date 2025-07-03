# AlexNet Implementation for SystemDS

This directory contains a comprehensive, modular implementation of AlexNet, the pioneering deep convolutional neural network introduced by Krizhevsky, Sutskever, and Hinton in 2012. Additionally, it includes the AlexNet-BN variant with batch normalization for large-batch training using LARS optimizer.

## Overview

AlexNet was the first deep CNN to significantly outperform traditional methods on ImageNet classification, marking a breakthrough in deep learning. Our implementation provides a flexible, reusable AlexNet architecture following SystemDS network conventions.

The implementation includes both the original AlexNet and the AlexNet-BN variant from "Large Batch Training of Convolutional Networks" (You et al., 2017), which enables stable training with large batch sizes using the LARS optimizer.

## Architecture

### Standard AlexNet Structure
- **Conv1**: 96 filters, 11×11, stride 4, pad 0 → ReLU → MaxPool 3×3, stride 2
- **Conv2**: 256 filters, 5×5, stride 1, pad 2 → ReLU → MaxPool 3×3, stride 2  
- **Conv3**: 384 filters, 3×3, stride 1, pad 1 → ReLU
- **Conv4**: 384 filters, 3×3, stride 1, pad 1 → ReLU
- **Conv5**: 256 filters, 3×3, stride 1, pad 1 → ReLU → MaxPool 3×3, stride 2
- **FC1**: 4096 neurons → ReLU → Dropout
- **FC2**: 4096 neurons → ReLU → Dropout
- **FC3**: num_classes neurons → Softmax

### AlexNet-BN Structure (Batch Normalization Variant)
- **Conv1**: 96 filters, 11×11, stride 4 → **BatchNorm** → ReLU → MaxPool 3×3, stride 2
- **Conv2**: 256 filters, 5×5, stride 1, pad 2 → **BatchNorm** → ReLU → MaxPool 3×3, stride 2  
- **Conv3**: 384 filters, 3×3, stride 1, pad 1 → **BatchNorm** → ReLU
- **Conv4**: 384 filters, 3×3, stride 1, pad 1 → **BatchNorm** → ReLU
- **Conv5**: 256 filters, 3×3, stride 1, pad 1 → **BatchNorm** → ReLU → MaxPool 3×3, stride 2
- **FC1**: 4096 neurons → ReLU → Dropout
- **FC2**: 4096 neurons → ReLU → Dropout
- **FC3**: num_classes neurons → Softmax

The AlexNet-BN variant adds batch normalization after each convolutional layer, enabling stable large-batch training with the LARS optimizer. This variant supports batch sizes up to 32K while maintaining convergence.

### Input/Output Specifications
- **Input**: 224×224×3 RGB images (ImageNet standard)
- **Output**: Configurable number of classes
- **Parameters**: ~60M parameters for 1000 classes

## Files

### Core Implementation
- `alexnet.dml` - Main AlexNet implementation with all functions

### Example Scripts
- `test_general_alexnet.dml` - Comprehensive test suite demonstrating all features

## Usage

### Basic Usage

#### Standard AlexNet
```dml
source("scripts/nn/networks/alexnet.dml") as alexnet

# Configuration
C = 3           # RGB channels
Hin = 224       # Input height
Win = 224       # Input width
num_classes = 10
seed = 42

# Initialize model
model = alexnet::init(C, Hin, Win, num_classes, seed)

# Forward pass
[predictions, cached_out] = alexnet::forward(X, C, Hin, Win, model, "train", 0.5)

# Backward pass
[dX, gradients] = alexnet::backward(dOut, cached_out, model, C, Hin, Win, 0.5)
```

#### AlexNet-BN with LARS Training
```dml
source("scripts/nn/networks/alexnet.dml") as alexnet

# Configuration for large-batch training
batch_size = 4096
use_bn = TRUE

# Get recommended hyperparameters
[base_lr, warmup_epochs, total_epochs] = alexnet::get_lars_hyperparams(batch_size, use_bn)

# Initialize AlexNet-BN model
[model, emas] = alexnet::init_with_bn(C, Hin, Win, num_classes, seed)

# Train with LARS
[trained_model, train_losses, val_accs] = alexnet::train_with_lars(
    X_train, Y_train, X_val, Y_val, C, Hin, Win, num_classes,
    total_epochs, batch_size, base_lr, use_bn, seed)
```

### Training Loop Example

```dml
# Training parameters
epochs = 10
batch_size = 64
lr = 0.01
weight_decay = 1e-4

# Initialize optimizer state (example with LARS)
lars_state = alexnet::init_lars_optim_params(model)

# Training loop
for (e in 1:epochs) {
  for (batch in batches) {
    # Forward pass
    [predictions, cached_out] = alexnet::forward(X_batch, C, Hin, Win, model, "train", 0.5)
    
    # Compute loss
    loss = alexnet::compute_loss(predictions, Y_batch, model, weight_decay)
    
    # Backward pass
    dOut = cross_entropy_loss::backward(predictions, Y_batch)
    [dX, gradients] = alexnet::backward(dOut, cached_out, model, C, Hin, Win, 0.5)
    
    # Update parameters with LARS
    [model, lars_state] = alexnet::update_params_with_lars(
        model, gradients, lr, 0.9, weight_decay, 0.001, lars_state)
  }
}
```

## API Reference

### Core Functions

#### `init(C, Hin, Win, num_classes, seed)`
Initialize AlexNet model parameters.

**Parameters:**
- `C`: Number of input channels (3 for RGB)
- `Hin`: Input height (224 for ImageNet)
- `Win`: Input width (224 for ImageNet)
- `num_classes`: Number of output classes
- `seed`: Random seed for initialization

**Returns:**
- `model`: List of initialized model parameters (16 matrices)

#### `forward(X, C, Hin, Win, model, mode, dropout_prob)`
Forward pass through the network.

**Parameters:**
- `X`: Input data, shape (N, C×Hin×Win)
- `C, Hin, Win`: Input dimensions
- `model`: Model parameters from `init()`
- `mode`: "train" or "test" (affects dropout)
- `dropout_prob`: Dropout probability (typically 0.5)

**Returns:**
- `out`: Predictions, shape (N, num_classes)
- `cached_out`: Cached intermediate outputs for backward pass

#### `backward(dOut, cached_out, model, C, Hin, Win, dropout_prob)`
Backward pass through the network.

**Parameters:**
- `dOut`: Gradient w.r.t. output, shape (N, num_classes)
- `cached_out`: Cached outputs from forward pass
- `model`: Model parameters
- `C, Hin, Win`: Input dimensions
- `dropout_prob`: Dropout probability used in forward pass

**Returns:**
- `dX`: Gradient w.r.t. input, shape (N, C×Hin×Win)
- `gradients`: List of gradients for all parameters

### AlexNet-BN Functions

#### `init_with_bn(C, Hin, Win, num_classes, seed)`
Initialize AlexNet-BN model parameters (with batch normalization).

**Parameters:**
- Same as `init()` function

**Returns:**
- `model`: List of model parameters including BN parameters (36 matrices)
- `emas`: List of exponential moving averages for BN layers

#### `forward_with_bn(X, C, Hin, Win, model, mode, dropout_prob)`
Forward pass through the AlexNet-BN network.

**Parameters:**
- Same as `forward()` function

**Returns:**
- `out`: Predictions, shape (N, num_classes)
- `cached_out`: Cached intermediate outputs for backward pass
- `emas_upd`: Updated exponential moving averages

#### `evaluate_with_bn(X, Y, C, Hin, Win, model, batch_size)`
Evaluate AlexNet-BN model on a dataset.

**Parameters:**
- Same as `evaluate()` function

**Returns:**
- `loss`: Average loss over the dataset
- `accuracy`: Classification accuracy

### LARS Training Utilities

#### `get_lars_hyperparams(batch_size, use_bn)`
Get recommended LARS hyperparameters based on batch size and network variant.

**Parameters:**
- `batch_size`: Training batch size
- `use_bn`: Whether using batch normalization

**Returns:**
- `base_lr`: Base learning rate (before batch scaling)
- `warmup_epochs`: Number of warmup epochs
- `total_epochs`: Recommended total training epochs

#### `get_lr_with_warmup(base_lr, epoch, iter, total_epochs, iters_per_epoch, batch_size, base_batch_size, warmup_epochs, decay_power)`
Learning rate scheduler with warmup, batch scaling, and polynomial decay.

**Parameters:**
- `base_lr`: Base learning rate
- `epoch`, `iter`: Current epoch and iteration
- `total_epochs`: Total training epochs
- `iters_per_epoch`: Iterations per epoch
- `batch_size`: Current batch size
- `base_batch_size`: Reference batch size (typically 256)
- `warmup_epochs`: Number of warmup epochs
- `decay_power`: Power for polynomial decay (typically 2)

**Returns:**
- `lr`: Scaled learning rate for current iteration

#### `train_with_lars(X_train, Y_train, X_val, Y_val, C, Hin, Win, num_classes, epochs, batch_size, base_lr, use_bn, seed)`
Train AlexNet with LARS optimizer following paper's best practices.

**Parameters:**
- `X_train`, `Y_train`: Training data and labels
- `X_val`, `Y_val`: Validation data and labels
- `C`, `Hin`, `Win`: Input dimensions
- `num_classes`: Number of output classes
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `base_lr`: Base learning rate (before batch scaling)
- `use_bn`: Whether to use batch normalization
- `seed`: Random seed

**Returns:**
- `model`: Trained model parameters
- `train_losses`: Training losses per epoch
- `val_accs`: Validation accuracies per epoch

### Optimizer Integration

The implementation provides seamless integration with multiple optimizers:

#### SGD
```dml
model_upd = alexnet::update_params_with_sgd(model, gradients, lr)
```

#### SGD with Momentum
```dml
momentum_state = alexnet::init_sgd_momentum_optim_params(model)
[model_upd, momentum_state_upd] = alexnet::update_params_with_sgd_momentum(
    model, gradients, lr, mu, momentum_state)
```

#### Adam
```dml
adam_state = alexnet::init_adam_optim_params(model)
[model_upd, adam_state_upd] = alexnet::update_params_with_adam(
    model, gradients, lr, beta1, beta2, epsilon, t, adam_state)
```

#### LARS (Layer-wise Adaptive Rate Scaling)
```dml
lars_state = alexnet::init_lars_optim_params(model)
[model_upd, lars_state_upd] = alexnet::update_params_with_lars(
    model, gradients, lr, mu, weight_decay, trust_coeff, lars_state)
```

### Utility Functions

#### `compute_loss(predictions, targets, model, weight_decay)`
Compute cross-entropy loss with L2 regularization.

#### `compute_accuracy(predictions, targets)`
Compute classification accuracy.

#### `evaluate(X, Y, C, Hin, Win, model, batch_size)`
Evaluate model on a dataset with batched processing.

## Advanced Features

### LARS Integration
This implementation includes full support for LARS (Layer-wise Adaptive Rate Scaling), enabling stable large-batch training:

- **Adaptive learning rates**: Different learning rates for different layers based on layer-wise norms
- **Trust coefficient**: Controls the adaptation strength (typically 0.001)
- **Weight decay support**: Built-in L2 regularization
- **Momentum**: Uses momentum for stable convergence
- **Batch scaling**: Linear learning rate scaling rule (LR = base_LR × batch_size / 256)
- **Warmup scheduling**: Linear warmup followed by polynomial decay
- **Large-batch support**: Stable training with batch sizes up to 32K (AlexNet-BN)

### Batch Normalization Benefits
The AlexNet-BN variant provides significant advantages for large-batch training:

- **Training stability**: BN normalizes activations, reducing internal covariate shift
- **Higher learning rates**: Enables aggressive learning rate scaling
- **Faster convergence**: Reduces the number of epochs needed for convergence
- **Better generalization**: Often improves final model accuracy
- **LARS synergy**: Works exceptionally well with LARS optimizer for large batches

### Modular Design
- **Clean separation**: Forward/backward passes are separate functions
- **Cacheable**: Intermediate outputs are cached for efficient backward pass
- **Extensible**: Easy to modify or extend the architecture
- **Compatible**: Follows SystemDS network conventions

### Memory Efficient
- **Batched evaluation**: Supports large datasets through batching
- **Flexible input sizes**: Supports different image resolutions
- **Optimized caching**: Minimal memory overhead for backward pass

## Performance Characteristics

### Memory Requirements
- **Model parameters**: ~240MB for 1000 classes (FP64)
- **Activation memory**: Scales with batch size
- **Recommended**: 8GB+ RAM for training with reasonable batch sizes

### Computational Complexity
- **Forward pass**: ~724M FLOPs for 224×224 input
- **Backward pass**: ~2.2B FLOPs (3× forward pass)
- **Training time**: Scales approximately linearly with batch size

## Testing

Run the comprehensive test suite:

```bash
./bin/systemds scripts/nn/examples/test_general_alexnet.dml
```

This verifies:
- Forward/backward pass correctness
- All optimizer integrations
- Loss computation
- Evaluation functions
- Memory efficiency

## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NIPS.

2. You, Y., Gitman, I., & Ginsburg, B. (2017). Large Batch Training of Convolutional Networks. arXiv preprint arXiv:1708.03888.

3. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML.

## Examples

See the following example scripts for complete usage:
- `scripts/nn/examples/test_general_alexnet.dml` - Feature verification
- `scripts/nn/examples/test_lars_vs_sgd.dml` - LARS comparison
- `scripts/nn/examples/Example-ImageNet_AlexNet_LARS_Demo.dml` - Quick demo
- `scripts/nn/examples/Example-AlexNet_BN_LARS.dml` - **AlexNet-BN with LARS training**

## License

Licensed under the Apache License, Version 2.0. See the main SystemDS LICENSE file for details. 