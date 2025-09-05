<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
{% end comment %}
-->

# Building the docs

This guide assume that you have cd'ed to `/src/main/python/docs/`.

## Requirements

To build the docs first install packages in `requires-docs.txt`

```bash
python3 -m pip install -r requires-docs.txt
```

## Make Docs

and then run `make html`:

```bash
make html
```

The docs will then be created at: `/src/main/python/docs/build/html/`.