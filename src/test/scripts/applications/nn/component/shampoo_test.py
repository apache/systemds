import numpy as np

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
    momentum = X * 0
    return(preconL, preconR, stepCounter, bufferL, bufferR, momentum, preconLInvPowerRoot, preconRInvPowerRoot, useDiag)

n = 5
m = 5
epsilon = 1e-4
lr = 0.005
diagThreshold = 10
rootEvery=10
preconEvery=10

# define weight matrix
X = np.array([
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

for diagThreshold in (1, 10):
    X_py = X.copy()
    dX = dX_main.copy()

    # preconL_py, preconR_py, useDiag_py = init_shampoo(X_py, epsilon, diagThreshold)
    # X_py, preconL_py, preconR_py = update_shampoo(X_py, dX, lr, preconL_py, preconR_py, useDiag_py)

    # preconL_py, preconR_py, momentum_py, useDiag_py = init_shampoo_momentum(X_py, epsilon, diagThreshold)
    # X_py, preconL_py, preconR_py, momentum_py = update_shampoo_momentum(X_py, dX, lr, preconL_py, preconR_py, momentum_py, useDiag_py)

    preconL, preconR, stepCounter, bufferL, bufferR, momentum, preconLInvPowerRoot, preconRInvPowerRoot, useDiag = init_shampoo_heuristic(X_py, epsilon, diagThreshold)
    X_py, preconL, preconR, momentum, stepCounter, bufferL, bufferR, preconLInvPowerRoot, preconRInvPowerRoot = update_shampoo_heuristic(X_py, dX, lr, preconL, preconR, momentum, stepCounter, rootEvery, preconEvery, bufferL, bufferR, preconLInvPowerRoot, preconRInvPowerRoot, useDiag)
    
    print("diagThreshold: " + str(diagThreshold))
    print(X_py)

 