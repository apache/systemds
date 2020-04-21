<!--
{% comment %}
Copyright 2020 Graz University of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# Build instructions

## Basic steps

The following steps have to be done for both cases

- Build SystemDS with maven first `mvn package -DskipTests`, with the working directory being `SYSTEMDS_ROOT` (Root directory of SystemDS)
- `cd` to this folder (basically `SYSTEMDS_ROOT/src/main/python`

### Building package

If we want to build the package for uploading to the repository via `python3 -m twine upload --repository-url [URL] dist/*` (will be automated in the future)

- Run `create_python_dist.py`

```bash
python3 create_python_dist.py
```

- now in the `./dist` directory there will exist the source distribution `systemds-VERSION.tar.gz` and the wheel distribution `systemds-VERSION-py3-none-any.whl`, with `VERSION` being the current version number
- Finished. We can now upload it with `python3 -m twine upload --repository-url [URL] dist/*`

### Building for development

If we want to build the package just locally for development, the following steps will suffice

- Run `pre_setup.py` (this will copy `lib` and `systemds-VERSION-SNAPSHOT.jar`)

```bash
python3 create_python_dist.py
```

- Finished. Test by running a test case of your choice:

```bash
python3 tests/test_matrix_binary_op.py
```
