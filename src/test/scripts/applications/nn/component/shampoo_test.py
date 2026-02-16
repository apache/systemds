#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import numpy as np
from systemds.context import SystemDSContext

# update shampoo 
def update_shampoo(X, dX, lr, preconL, preconR, useDiag):
    if(not useDiag):

        preconL = preconL + dX @ dX.T
        preconR = preconR + dX.T @dX

        LEigenvalue, LEigenvector = np.linalg.eig(preconL)
        preconLInvPowerRoot = LEigenvector @ np.diag(LEigenvalue**(-0.25)) @ LEigenvector.T

        REigenvalue, REigenvector = np.linalg.eig(preconR)
        preconRInvPowerRoot = REigenvector @ np.diag(REigenvalue**(-0.25)) @ REigenvector.T

        X = X - lr * preconLInvPowerRoot @ dX @ preconRInvPowerRoot
  
  # Diagonal Shampoo:
  # Memory-efficient approximation for large parameter matrices
    else:
        n = dX.shape[0]
        m = dX.shape[1]

        preconL = preconL + (dX**2).sum(axis=1, keepdims=True)          
        preconR = preconR + (dX**2).sum(axis=0, keepdims=True)

        preconLScale = preconL**(-0.25)        
        preconRScale = preconR**(-0.25)

        preconLMatrix = preconLScale @ np.ones(shape=[1, m])
        preconRMatrix = np.ones(shape=(n, 1)) @ preconRScale

        scaledGrad = dX * preconLMatrix
        scaledGrad = scaledGrad * preconRMatrix

        X = X - lr * scaledGrad

    return(X, preconL, preconR)

# init shampoo

def init_shampoo(X, epsilon, useDiagThreshold):
    if((X.shape[0] > useDiagThreshold) or (X.shape[1] > useDiagThreshold)):
        preconL = np.full(shape=(X.shape[0], 1), fill_value=epsilon, dtype=np.float64)
        preconR = np.full(shape=(1, X.shape[1]), fill_value=epsilon, dtype=np.float64)
        useDiag = True
    else:
        preconL = np.eye(X.shape[0], dtype=np.float64) * epsilon
        preconR = np.eye(X.shape[1], dtype=np.float64) * epsilon

        useDiag = False
    return(preconL, preconR, useDiag)

# update shampoo 
def update_shampoo_momentum(X, dX, lr, preconL, preconR, momentum, useDiag):
    momentum = 0.9 * momentum + (0.1)*dX
    if(not useDiag):

        preconL = preconL + dX @ dX.T
        preconR = preconR + dX.T @dX

        LEigenvalue, LEigenvector = np.linalg.eig(preconL)
        preconLInvPowerRoot = LEigenvector @ np.diag(LEigenvalue**(-0.25)) @ LEigenvector.T

        REigenvalue, REigenvector = np.linalg.eig(preconR)
        preconRInvPowerRoot = REigenvector @ np.diag(REigenvalue**(-0.25)) @ REigenvector.T

        X = X - lr * preconLInvPowerRoot @ momentum @ preconRInvPowerRoot
  
  # Diagonal Shampoo:
  # Memory-efficient approximation for large parameter matrices
    else:
        n = dX.shape[0]
        m = dX.shape[1]

        preconL = preconL + (dX**2).sum(axis=1, keepdims=True)          
        preconR = preconR + (dX**2).sum(axis=0, keepdims=True)

        preconLScale = preconL**(-0.25)        
        preconRScale = preconR**(-0.25)

        preconLMatrix = preconLScale @ np.ones(shape=[1, m])
        preconRMatrix = np.ones(shape=(n, 1)) @ preconRScale

        scaledGrad = momentum * preconLMatrix
        scaledGrad = scaledGrad * preconRMatrix

        X = X - lr * scaledGrad

    return(X, preconL, preconR, momentum)

# init shampoo

def init_shampoo_momentum(X, epsilon, useDiagThreshold):
    if((X.shape[0] > useDiagThreshold) or (X.shape[1] > useDiagThreshold)):
        preconL = np.full(shape=(X.shape[0], 1), fill_value=epsilon, dtype=np.float64)
        preconR = np.full(shape=(1, X.shape[1]), fill_value=epsilon, dtype=np.float64)
        useDiag = True
    else:
        preconL = np.eye(X.shape[0], dtype=np.float64) * epsilon
        preconR = np.eye(X.shape[1], dtype=np.float64) * epsilon

        useDiag = False

    momentum = X * 0
    return(preconL, preconR, momentum, useDiag)

# update shampoo 
def update_shampoo_heuristic(X, dX, lr, preconL, preconR, momentum, stepCounter, rootEvery, preconEvery, bufferL, bufferR, preconLInvPowerRoot, preconRInvPowerRoot, useDiag):
    momentum = 0.9 * momentum + (0.1)*dX
    if(not useDiag):
        bufferL = bufferL + (dX @ dX.T)
        bufferR = bufferR + (dX.T @dX)

        if ((stepCounter > 0) and (stepCounter % preconEvery == 0)):
            preconL = preconL + bufferL
            preconR = preconR + bufferR
            bufferL = bufferL * 0
            bufferR = bufferR * 0

        if ((stepCounter > 0) and (stepCounter % rootEvery == 0)):
            LEigenvalue, LEigenvector = np.linalg.eig(preconL)
            preconLInvPowerRoot = LEigenvector @ np.diag(LEigenvalue**(-0.25)) @ LEigenvector.T

            REigenvalue, REigenvector = np.linalg.eig(preconR)
            preconRInvPowerRoot = REigenvector @ np.diag(REigenvalue**(-0.25)) @ REigenvector.T

        X = X - lr * preconLInvPowerRoot @ momentum @ preconRInvPowerRoot
  
  # Diagonal Shampoo:
  # Memory-efficient approximation for large parameter matrices
    else:
        n = dX.shape[0]
        m = dX.shape[1]

        bufferL = bufferL + (dX**2).sum(axis=1, keepdims=True)    
        bufferR = bufferR + (dX**2).sum(axis=0, keepdims=True)

        if ((stepCounter > 0) and (stepCounter % preconEvery == 0)):
            preconL = preconL +  bufferL     
            preconR = preconR + bufferR
            bufferL = bufferL * 0
            bufferR = bufferR * 0
        
        if ((stepCounter > 0) and (stepCounter % rootEvery == 0)):
            preconLInvPowerRoot = preconL**(-0.25)        
            preconRInvPowerRoot = preconR**(-0.25)

        preconLMatrix = preconLInvPowerRoot @ np.ones(shape=[1, m])
        preconRMatrix = np.ones(shape=(n, 1)) @ preconRInvPowerRoot

        scaledGrad = momentum * preconLMatrix
        scaledGrad = scaledGrad * preconRMatrix

        X = X - lr * scaledGrad

    return(X, preconL, preconR, momentum, stepCounter, bufferL, bufferR, preconLInvPowerRoot, preconRInvPowerRoot)

# init shampoo

def init_shampoo_heuristic(X, epsilon, useDiagThreshold):
    if((X.shape[0] > useDiagThreshold) or (X.shape[1] > useDiagThreshold)):
        preconL = np.full(shape=(X.shape[0], 1), fill_value=epsilon, dtype=np.float64)
        preconR = np.full(shape=(1, X.shape[1]), fill_value=epsilon, dtype=np.float64)
        preconLInvPowerRoot = preconL**(-0.25)
        preconRInvPowerRoot = preconR**(-0.25)
        useDiag = True
    else:
        preconL = np.eye(X.shape[0], dtype=np.float64) * epsilon
        preconR = np.eye(X.shape[1], dtype=np.float64) * epsilon
        preconLInvPowerRoot = np.eye(X.shape[0], dtype=np.float64) * epsilon**(-0.25)
        preconRInvPowerRoot = np.eye(X.shape[1], dtype=np.float64) * epsilon**(-0.25)

        useDiag = False

    momentum = X * 0
    bufferR = preconR * 0
    bufferL = preconL * 0
    stepCounter = 0
    return(preconL, preconR, stepCounter, bufferL, bufferR, momentum, preconLInvPowerRoot, preconRInvPowerRoot, useDiag)



# define weight matrix
X_main = np.array([
    [ 0.12, -0.45,  0.33,  0.08, -0.19],
    [-0.27,  0.41, -0.05,  0.22,  0.14],
    [ 0.09, -0.31,  0.26, -0.48,  0.37],
    [ 0.44,  0.06, -0.29,  0.15, -0.11],
    [-0.38,  0.24,  0.17, -0.07,  0.52],
], dtype=np.float64)

# define gradient
dX_main = np.array([
    [ 0.015, -0.022,  0.008,  0.031, -0.012],
    [-0.009,  0.027, -0.014,  0.005,  0.019],
    [ 0.021, -0.006,  0.011, -0.025,  0.004],
    [-0.018,  0.013, -0.029,  0.007, -0.016],
    [ 0.010, -0.017,  0.024, -0.003,  0.028],
], dtype=np.float64)

with SystemDSContext() as sds:
    n = 5
    m = 5
    epsilon = 1e-4
    lr = 0.005
    diagThreshold = 10
    rootEvery=10
    preconEvery=10
    
    shampooUtils = sds.source("src/test/scripts/applications/nn/component/shampoo_test2.dml", "shampooUtils")

    for shampoo_type in ["shampoo", "shampoo_momentum", "shampoo_heuristic"]:
        for diagThreshold in (1, 10):
            X = X_main.copy()
            dX = dX_main.copy()
            
            if shampoo_type=="shampoo":
                preconL_py, preconR_py, useDiag_py = init_shampoo(X, epsilon, diagThreshold)
                X_py, preconL_py, preconR_py = update_shampoo(X, dX, lr, preconL_py, preconR_py, useDiag_py)
            elif shampoo_type=="shampoo_momentum":
                preconL_py, preconR_py, momentum_py, useDiag_py = init_shampoo_momentum(X, epsilon, diagThreshold)
                X_py, preconL_py, preconR_py, momentum_py = update_shampoo_momentum(X, dX, lr, preconL_py, preconR_py, momentum_py, useDiag_py)
            elif shampoo_type=="shampoo_heuristic":
                preconL, preconR, stepCounter, bufferL, bufferR, momentum, preconLInvPowerRoot, preconRInvPowerRoot, useDiag = init_shampoo_heuristic(X, epsilon, diagThreshold)
                X_py, preconL, preconR, momentum, stepCounter, bufferL, bufferR, preconLInvPowerRoot, preconRInvPowerRoot = update_shampoo_heuristic(X, dX, lr, preconL, preconR, momentum, stepCounter, rootEvery, preconEvery, bufferL, bufferR, preconLInvPowerRoot, preconRInvPowerRoot, useDiag)
            
            X_updated = shampooUtils.test_update(sds.from_numpy(X), sds.from_numpy(dX), sds.scalar(shampoo_type), epsilon, lr, diagThreshold, rootEvery, preconEvery).compute()
            X_updated = np.asarray(X_updated, dtype=np.float64)

            identical = np.allclose(X_py, X_updated, rtol=1e-7, atol=1e-9)

            diagonal = "diagonal" if diagThreshold==1 else "non-diagonal"

            print(f"{shampoo_type} - {diagonal}: {'Test Passed' if identical else 'Test Failed'}")



