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

__all__ = ['normal', 'uniform', 'poisson']

from ..defmatrix import *

# Special object used internally to specify the placeholder which will be replaced by output ID
# This helps to provide dml containing output ID in constructSamplingNode
OUTPUT_ID = '$$OutputID$$'

def constructSamplingNode(inputs, dml):
    """
    Convenient utility to create an intermediate of AST.

    Parameters
    ----------
    inputs = list of input matrix objects and/or DMLOp
    dml = list of DML string (which will be eventually joined before execution). To specify out.ID, please use the placeholder
    """
    dmlOp = DMLOp(inputs)
    out = matrix(None, op=dmlOp)
    dmlOp.dml = [out.ID if x==OUTPUT_ID else x for x in dml]
    return out

INPUTS = []
def asStr(arg):
    """
    Internal use only: Convenient utility to update inputs and return appropriate string value
    """
    if isinstance(arg, matrix):
        INPUTS = INPUTS + [ arg ]
        return arg.ID
    else:
        return str(arg)
        
def normal(loc=0.0, scale=1.0, size=(1,1), sparsity=1.0):
    """
    Draw random samples from a normal (Gaussian) distribution.
        
    Parameters
    ----------
    loc: Mean ("centre") of the distribution.
    scale: Standard deviation (spread or "width") of the distribution.
    size: Output shape (only tuple of length 2, i.e. (m, n), supported).
    sparsity: Sparsity (between 0.0 and 1.0).
    
    Examples
    --------

    >>> import systemml as sml
    >>> import numpy as np
    >>> sml.setSparkContext(sc)
    >>> from systemml import random
    >>> m1 = sml.random.normal(loc=3, scale=2, size=(3,3))
    >>> m1.toNumPy()
    array([[ 3.48857226,  6.17261819,  2.51167259],
           [ 3.60506708, -1.90266305,  3.97601633],
           [ 3.62245706,  5.9430881 ,  2.53070413]])
    
    """
    if len(size) != 2:
        raise TypeError('Incorrect type for size. Expected tuple of length 2')
    INPUTS = []
    rows = asStr(size[0])
    cols = asStr(size[1])
    loc = asStr(loc)
    scale = asStr(scale)
    sparsity = asStr(sparsity)
    # loc + scale*standard normal
    return constructSamplingNode(INPUTS, [OUTPUT_ID, ' = ',  loc,' + ',  scale,' * random.normal(', rows, ',', cols, ',',  sparsity, ')\n'])

def uniform(low=0.0, high=1.0, size=(1,1), sparsity=1.0):
    """
    Draw samples from a uniform distribution.
        
    Parameters
    ----------
    low: Lower boundary of the output interval.
    high: Upper boundary of the output interval.
    size: Output shape (only tuple of length 2, i.e. (m, n), supported). 
    sparsity: Sparsity (between 0.0 and 1.0).

    Examples
    --------
    
    >>> import systemml as sml
    >>> import numpy as np
    >>> sml.setSparkContext(sc)
    >>> from systemml import random
    >>> m1 = sml.random.uniform(size=(3,3))
    >>> m1.toNumPy()
    array([[ 0.54511396,  0.11937437,  0.72975775],
           [ 0.14135946,  0.01944448,  0.52544478],
           [ 0.67582422,  0.87068849,  0.02766852]])

    """
    if len(size) != 2:
        raise TypeError('Incorrect type for size. Expected tuple of length 2')
    INPUTS = []
    rows = asStr(size[0])
    cols = asStr(size[1])
    low = asStr(low)
    high = asStr(high)
    sparsity = asStr(sparsity)
    return constructSamplingNode(INPUTS, [OUTPUT_ID, ' = random.uniform(', rows, ',', cols, ',',  sparsity, ',',  low, ',',  high, ')\n'])

def poisson(lam=1.0, size=(1,1), sparsity=1.0):
    """
    Draw samples from a Poisson distribution.
    
    Parameters
    ----------
    lam: Expectation of interval, should be > 0.
    size: Output shape (only tuple of length 2, i.e. (m, n), supported). 
    sparsity: Sparsity (between 0.0 and 1.0).
    
    
    Examples
    --------
    
    >>> import systemml as sml
    >>> import numpy as np
    >>> sml.setSparkContext(sc)
    >>> from systemml import random
    >>> m1 = sml.random.poisson(lam=1, size=(3,3))
    >>> m1.toNumPy()
    array([[ 1.,  0.,  2.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
           
    """
    if len(size) != 2:
        raise TypeError('Incorrect type for size. Expected tuple of length 2')
    INPUTS = []
    rows = asStr(size[0])
    cols = asStr(size[1])
    lam = asStr(lam)
    sparsity = asStr(sparsity)
    return constructSamplingNode(INPUTS, [OUTPUT_ID, ' = random.poisson(', rows, ',', cols, ',',  sparsity, ',',  lam, ')\n'])
