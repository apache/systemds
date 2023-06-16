from systemds.operator import Matrix

def affine(X: Matrix,
           W: Matrix,
           b: Matrix):
    """
        Computes the forward pass for an affine layer
        with M neurons.  The input data has N examples, each with D
        features.
        :param X: An nxd matrix containing input values from the previous layer
        :param W: An dxm weight matrix
        :param b: An 1xm matrix containing the bias
        :return: An nxd matrix
    """
    res = X.sds_context.source("../../../../../scripts/nn/layers/affine.dml", "forward").forward(X, W, b)
    return res