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
{% end comment %}
-->

# onnx-systemds

A tool for importing/exporting [onnx](https://github.com/onnx/onnx/blob/master/docs/IR.md) graphs into/from SystemDS DML scripts.

For a more detailed description of this converter refer to the [description of the converter design](docs/onnx-systemds-design.md)

## Prerequisites

to run onnx-systemds you need:

*  [onnx](https://github.com/onnx/onnx): [Installation instructions](https://github.com/onnx/onnx#installation)
* You need to [set up the environment](../../../../../bin/README.md)

## Usage

An example call from the `src/main/python` directory of systemds:

```bash
python -m systemds.onnx_systemds.convert tests/onnx/test_models/simple_mat_add.onnx
```

This will generate the dml script `simple_mat_add.dml` in your current directory. 

### Run Tests

Form the `src/main/python` directory of systemds:

At first generate the test models:

```bash
python tests/onnx/test_models/model_generate.py
```

Then you can run the tests:

```bash
python -m unittest tests/onnx/test_simple.py
```
