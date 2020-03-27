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

# Building the docs

To build the docs install packages in `requires-docs.txt` and then run `make html`:

```bash
python3 -m pip install -r requires-docs.txt
make html
```

The docs will be placed in the `./build` directory.
