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

from .defmatrix import *
from . import MLContext, pydml

from dill.source import getsource
from textwrap import dedent

__all__ = ['parallelize']

def parallelize(fn):
    """
    The current implementation can only pushdown subset of Python that maps to PyDML.
    Also, the inputs and outputs need to be explicitly specified.
    We will relax this constraints in future version by performing AST analysis.
    """
    def inner(*args, **kwargs):
        if matrix.ml is None:
            raise Exception('Expected setSparkContext(sc) to be called.')
        # ------------------------------------------------------------------------------------
        # Assumption that @parallelize and function definition has one line. Fix this by having a class that extends ast.NodeVisitor
        if 'inputs' not in kwargs or 'outputs' not in kwargs:
            raise ValueError('The function to parallelize should contains exactly two named arguments with keys: inputs, outputs')
        codeWithParallelizeHeader = getsource(fn)
        # ------------------------------------------------------------------------------------
        codeWithoutParallelizeHeader = ''.join(['\n', dedent(codeWithParallelizeHeader.split("\n",2)[2]), '\n'])
        # print "Executing code as pydml:\n" + codeWithoutParallelizeHeader
        script = pydml(codeWithoutParallelizeHeader)
        for name, value in kwargs['inputs'].items():
            if isinstance(value, matrix):
                # TODO: Uncomment this after _py2java allow NumPy array
                # value.eval()
                # script.input(name, value.data)
                script.input(name, value.toDataFrame())
            else:
                script.input(name, value)
        map(lambda o: script.output(o), kwargs['outputs'])        
        matrix.ml = MLContext(matrix.sc)
        results = matrix.ml.execute(script)
        return map(lambda o: results.get(o), kwargs['outputs'])
    return inner

        