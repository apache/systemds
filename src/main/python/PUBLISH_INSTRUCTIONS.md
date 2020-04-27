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

# Publishing Instructions

## Building SystemDS jar (with dependency jars)

The following steps have to be done for both the cases

- Build SystemDS with maven first `mvn package -P distribution`, with the working 
  directory being `SYSTEMDS_ROOT` (Root directory of SystemDS)
- `cd` to this folder (basically `SYSTEMDS_ROOT/src/main/python`)

## Building python package

- Run `create_python_dist.py`

```bash
python3 create_python_dist.py
```

- now in the `./dist` directory there will exist the source distribution `systemds-VERSION.tar.gz` 
  and the wheel distribution `systemds-VERSION-py3-none-any.whl`, with `VERSION` being the current version number

## Publishing package

If we want to build the package for uploading to the repository via `python3 -m twine upload dist/*` 
  (will be automated in the future)

- Install twine with `pip install --upgrade twine`

- Follow the instructions from the [Guide](https://packaging.python.org/tutorials/packaging-projects/)
    1. Create an API-Token in the account (leave the page open or copy the token, it will only be shown once)
    2. Execute the command `python3 -m twine upload dist/*`
        - Optional: `pip install keyrings.alt`(use with caution!) if you get `UserWarning: No recommended backend was available.`
    3. Username is `__token__`
    4. Password is the created API-Token **with** `pypi-` prefix
