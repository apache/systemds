<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

# SystemML-NN

### A deep learning library for [Apache SystemML](https://github.com/apache/systemml).

## Examples:
#### Please see the [`examples`](examples) folder for more detailed examples, or view the following two quick examples.
### Neural net for regression with vanilla SGD:
```python
# Imports
source("nn/layers/affine.dml") as affine
source("nn/layers/l2_loss.dml") as l2_loss
source("nn/layers/relu.dml") as relu
source("nn/optim/sgd.dml") as sgd

# Generate input data
N = 1024 # num examples
D = 100 # num features
t = 1 # num targets
X = rand(rows=N, cols=D, pdf="normal")
y = rand(rows=N, cols=t)

# Create 2-layer network:
## affine1 -> relu1 -> affine2
M = 64 # number of neurons
[W1, b1] = affine::init(D, M)
[W2, b2] = affine::init(M, t)

# Initialize optimizer
lr = 0.05  # learning rate
mu = 0.9  # momentum
decay = 0.99  # learning rate decay constant

# Optimize
print("Starting optimization")
batch_size = 32
epochs = 5
iters = 1024 / batch_size
for (e in 1:epochs) {
  for(i in 1:iters) {
    # Get next batch
    X_batch = X[i:i+batch_size-1,]
    y_batch = y[i:i+batch_size-1,]

    # Compute forward pass
    out1 = affine::forward(X_batch, W1, b1)
    outr1 = relu::forward(out1)
    out2 = affine::forward(outr1, W2, b2)

    # Compute loss
    loss = l2_loss::forward(out2, y_batch)
    print("L2 loss: " + loss)

    # Compute backward pass
    dout2 = l2_loss::backward(out2, y_batch)
    [doutr1, dW2, db2] = affine::backward(dout2, outr1, W2, b2)
    dout1 = relu::backward(doutr1, out1)
    [dX_batch, dW1, db1] = affine::backward(dout1, X_batch, W1, b1)

    # Optimize with vanilla SGD
    W1 = sgd::update(W1, dW1, lr)
    b1 = sgd::update(b1, db1, lr)
    W2 = sgd::update(W2, dW2, lr)
    b2 = sgd::update(b2, db2, lr)
  }
  # Decay learning rate
  lr = lr * decay
}
```

### Neural net for multi-class classification with dropout and SGD w/ Nesterov momentum:
```python
# Imports
source("nn/layers/affine.dml") as affine
source("nn/layers/cross_entropy_loss.dml") as cross_entropy_loss
source("nn/layers/dropout.dml") as dropout
source("nn/layers/relu.dml") as relu
source("nn/layers/softmax.dml") as softmax
source("nn/optim/sgd_nesterov.dml") as sgd_nesterov

# Generate input data
N = 1024 # num examples
D = 100 # num features
t = 5 # num targets
X = rand(rows=N, cols=D, pdf="normal")
classes = round(rand(rows=N, cols=1, min=1, max=t, pdf="uniform"))
y = matrix(0, rows=N, cols=t)
parfor (i in 1:N) {
  y[i, as.scalar(classes[i,1])] = 1  # one-hot encoding
}

# Create network:
# affine1 -> relu1 -> dropout1 -> affine2 -> relu2 -> dropout2 -> affine3 -> softmax
H1 = 64 # number of neurons in 1st hidden layer
H2 = 64 # number of neurons in 2nd hidden layer
p = 0.5  # dropout probability
[W1, b1] = affine::init(D, H1)
[W2, b2] = affine::init(H1, H2)
[W3, b3] = affine::init(H2, t)

# Initialize SGD w/ Nesterov momentum optimizer
lr = 0.05  # learning rate
mu = 0.5  # momentum
decay = 0.99  # learning rate decay constant
vW1 = sgd_nesterov::init(W1); vb1 = sgd_nesterov::init(b1)
vW2 = sgd_nesterov::init(W2); vb2 = sgd_nesterov::init(b2)
vW3 = sgd_nesterov::init(W3); vb3 = sgd_nesterov::init(b3)

# Optimize
print("Starting optimization")
batch_size = 64
epochs = 10
iters = 1024 / batch_size
for (e in 1:epochs) {
  for(i in 1:iters) {
    # Get next batch
    X_batch = X[i:i+batch_size-1,]
    y_batch = y[i:i+batch_size-1,]

    # Compute forward pass
    ## layer 1:
    out1 = affine::forward(X_batch, W1, b1)
    outr1 = relu::forward(out1)
    [outd1, maskd1] = dropout::forward(outr1, p, -1)
    ## layer 2:
    out2 = affine::forward(outd1, W2, b2)
    outr2 = relu::forward(out2)
    [outd2, maskd2] = dropout::forward(outr2, p, -1)
    ## layer 3:
    out3 = affine::forward(outd2, W3, b3)
    probs = softmax::forward(out3)

    # Compute loss
    loss = cross_entropy_loss::forward(probs, y_batch)
    print("Cross entropy loss: " + loss)

    # Compute backward pass
    ## loss:
    dprobs = cross_entropy_loss::backward(probs, y_batch)
    ## layer 3:
    dout3 = softmax::backward(dprobs, out3)
    [doutd2, dW3, db3] = affine::backward(dout3, outd2, W3, b3)
    ## layer 2:
    doutr2 = dropout::backward(doutd2, outr2, p, maskd2)
    dout2 = relu::backward(doutr2, out2)
    [doutd1, dW2, db2] = affine::backward(dout2, outd1, W2, b2)
    ## layer 1:
    doutr1 = dropout::backward(doutd1, outr1, p, maskd1)
    dout1 = relu::backward(doutr1, out1)
    [dX_batch, dW1, db1] = affine::backward(dout1, X_batch, W1, b1)

    # Optimize with SGD w/ Nesterov momentum
    [W1, vW1] = sgd_nesterov::update(W1, dW1, lr, mu, vW1)
    [b1, vb1] = sgd_nesterov::update(b1, db1, lr, mu, vb1)
    [W2, vW2] = sgd_nesterov::update(W2, dW2, lr, mu, vW2)
    [b2, vb2] = sgd_nesterov::update(b2, db2, lr, mu, vb2)
    [W3, vW3] = sgd_nesterov::update(W3, dW3, lr, mu, vW3)
    [b3, vb3] = sgd_nesterov::update(b3, db3, lr, mu, vb3)
  }
  # Anneal momentum towards 0.999
  mu = mu + (0.999 - mu)/(1+epochs-e)
  # Decay learning rate
  lr = lr * decay
}
```
