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

# Python Test

Tests are easily executed using unittest:

But before executing the tests it is recommended to go through systemds [Setting SYSTEMDS_ROOT environment](/bin/README.md)

```bash
# Single thread:
python -m unittest discover -s tests -p 'test_*.py'

# Parallel
unittest-parallel -t . -s tests --module-fixtures
```

This command searches through the test directory and finds all python files starting with `test_` and executes them.

The only tests not executed using the above commands are `Federated Tests`.

## Federated Tests

To execute the Federated Tests, use:

```bash
./tests/federated/runFedTest.sh
```

Federated experiments are a little different from the rest, since they require some setup in form of federated workers.

See more details in the [script](federated/runFedTest.sh)

https://github.com/nttcslab/byol-a/blob/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d512.pth