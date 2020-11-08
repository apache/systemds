.. -------------------------------------------------------------
..
.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.
..
.. -------------------------------------------------------------

Design
======

This document describes the initial design of `onnx-systemds`

For dealing with different operator-set versions of onnx the current strategy is to use the
`converter provided by onnx <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx>`_ to convert to a common version.

However, the converter does not support adapters for all op-sets/operators so this conversion will fail for many models.
On the onnx repository you can find a list of
`currently supported adapters <https://github.com/onnx/onnx/blob/master/onnx/version_converter.py#L21>`_


Goals
-----

 - Support for importing `operators of the ONNX base definition <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
 - Support for importing `operators defined by ONNX-ML <https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md>`_
 - Support for exporting DML script to ONNX graphs


Limitations
------------

 - Not able to support all data types / operators as they are not currently supported by SystemDS

Onnx - Operators
-----------------

Onnx includes several very simple and also more complex operators.
When implementing an operator it's best to have a look at the
`operator schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_,
which precisely define the inputs, outputs and attributes of the operation.

Besides the standard onnx definition, there also exists onnx-ML the operator schemas for which are defined in a
`separate document <https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md>`_.
It is an extension of the standard onnx format, however currently only onnx standard operators are supported.

Onnx - Files
-------------

Onnx uses the `ProtoBuf format <https://developers.google.com/protocol-buffers/>`_.
It specifies this representation in several ``.proto``/``.proto3``
`files <https://github.com/onnx/onnx/tree/master/onnx>`_ again with dedicated files for onnx-ML.
These files are helpful to understand the underlying structure and values that are possible.

Protobuf creates the underlying structure such that you can access elements of the onnx graph as if they were
class members. For more information take a look at
`Google's protocol-buffer documentation
<https://developers.google.com/protocol-buffers/docs/pythontutorial#the-protocol-buffer-api>`_.

This is also why in its current form, this converter does not convert the protobuf-structure into an internal format,
as the provided protobuf structure can already be conveniently used. Instead,
there exist a number of onnx-helper functions/classes (see ``onnx_helper.py``).

Traversing the Graph
---------------------

For creating the script, it is essential to insert computations in the right order into the dml-script.
To do this, the converter builds a tree-structure (DAG) from the protobuf-nodes
(see `render.gen_graph_functions`).

 - For traversing the graph, we start from the bottom.
 - The converter starts with the graph-outputs as available outputs.
 - It generates the dml snippets in reverse-order

Graph traversal
^^^^^^^^^^^^^^^^

1. Find a node for which all outputs are available.

2. Process the node:

    - Generate the dml parts for this node
    - add its inputs to the list of available outputs
    - remove the node from the graph

3. if there are nodes left restart at 1.

Example
^^^^^^^

In the example below with the nodes ``Add``, ``MatMul`` and ``Sub``, we would start with ``F`` as available output.
Therefore the first node to insert would be ``Sub``. After inserting ``Sub`` its inputs become available outputs,
therefore all outputs of ``MatMul`` become available. Finally, after removing ``MatMul`` from the graph all outputs
to ``Add`` are available, and it can be removed from the graph as well.

.. image:: /assets/sample_graph.png
    :width: 200px
    :align: center
    :alt: sample graph


Rendering DML scripts
---------------------

The main idea of this converter is, that the logic for generating the actual dml-syntax is handled by
`Jinja templates <https://jinja.palletsprojects.com/en/2.11.x/>`_ (located in ``/templates``).
Therefore the python code stays uncluttered, because it does not have to merge strings together to produce valid
dml-syntax and instead simply provides the elements that are needed to render the script.

The template-engine then takes these inputs and renders a human readable script with valid dml syntax.
To improve readability the generator also automatically ads the doc-strings which are part of the onnx-definitions as
comments to the script.

When traversing the graph, a script part is generated for each node consisting of three elements:

 - `dml_script` The actual script snipped for the node
 - `imports` Imports required for the node
 - `sub_graphs` Any sub_graphs of the node that need to be handled

The function that is called for rendering a specific operator is defined in the dictionary
``operator_generators`` in ``render.py``

1. `dml_script`
^^^^^^^^^^^^^^^^^^

Depending on the operator this can be a function call or a more complex dml-snippet.
This part is generated by the template-engine when the corresponding template is rendered.

Many onnx-operators can be handled by a single template file. There exists a ``function_call.dml.jinja``
template which should be able to handle a large number of operators.

2. `imports`
^^^^^^^^^^^^^

Some operators are handled by calling scripts provided by systemds located in ``$SYSTEMDS_ROOT/scripts``.
To enable these imports, the converter automatically resolves the ``$SYSTEMDS_ROOT``
environment variable and adds a ``setw($SYSTEMDS_ROOT/scripts)`` to the script.

3. `sub_graphs`
^^^^^^^^^^^^^^^^^

Since sub-graphs have their own variable scope and are independent, they are handled as separate functions.
The converter generates a function for each graph in the model.
In the main-graph, the sub-graph is replaced by a function call to the sub-graph function.
To handle this the function ``render.gen_graph_functions`` recursively calls itself to render sub-graph functions
(and also the sub-graph functions of sub-graphs and so on...).

Final Script
------------

In the final render all required imports, the sub-functions and the main-function are combined in a single dml-file.

Implementing new operators
----------------------------

When implementing an operator it's best to have a look at the
`operator schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
which exactly define the inputs, outputs and attributes of the operation

It is also nice to have a test-model to work with, to generate one refer to
``tests/onnx/test_models/model_generate.py``.

To implement a new operator, the function that handles the operator needs to be defined in the ``operator_generators``
located in ``render.py``.
All functions listed in this dictionary need to have the same call structure.

If there exists a dml-script (in ``$SYSTEMDS_ROOT/scripts``) that provides the functionality the operator
can be implemented by translating the arguments/inputs, adding the import-render and function-call-render to this script.

Testing models
---------------

onnx provides a convenient way for
`creating models <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#checking-an-onnx-model>`_
using helper functions in python. All current test-models are produced like this (see ``tests/onnx/test_models``).

Creating a Testcase
^^^^^^^^^^^^^^^^^^^^^

The current test-system takes a model, converts it to dml using the converter and then runs a
``dml_wrapper`` which calls the model-function using the script ``$SYSTEMDS_ROOT/bin/systemds``.
Finally, the output (stored by the dml-wrapper) is compared to a reference output.

When creating files stick to the naming conventions of other files in the same folder.

Steps:
""""""""

1. Create a model in ``tests/onnx/test_models``, e.g. ``sample_model.onnx``

2. Create a dml wrapper that calls the model-function in ``tests/onnx/dml_wrapper/sample_model_wrapper.dml``

    - The wrapper needs to call the model-function and store the output to ``output_test/sample_model.out``
    - The name of the model-function is generated from the model-name (see ``util.generate_function_name`` )

3. Provide a reference output in ``tests/onnx/output_reference/sample_model_reference.out``

4. Create the unit test function.

Tools
------

 - `Pycharm <https://www.jetbrains.com/pycharm/>`_ in the professional version allows you to  `debug template files <https://www.jetbrains.com/help/pycharm/templates.html#debug>`_ which can be handy
 - `Neutron <https://github.com/lutzroeder/netron>`_ is a nice free tool for viewing onnx-graphs