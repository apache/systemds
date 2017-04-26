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

# SystemML-NN Tests

#### This folder contains tests for the *SystemML-NN* (`nn`) deep learning library.

---
## Tests
#### All layers are tested for correct derivatives ("gradient-checking"), and many layers also have correctness tests against simpler reference implementations.
* `grad_check.dml` - Contains gradient-checks for all layers as individual DML functions.
* `test.dml` - Contains correctness tests for several of the more complicated layers by checking against simple reference implementations, such as `conv_simple.dml`.  All tests are formulated as individual DML functions.
* `tests.dml` - A DML script that runs all of the tests in `grad_check.dml` and `test.dml`.

## Execution
* `spark-submit SystemML.jar -f nn/test/tests.dml` from the base of the project.
