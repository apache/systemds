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

# Contributing to SystemDS

Thanks for taking the time to contribute to SystemDS!

The following are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

___
### Contribution Guidelines and Standards

Before contributing a pull request for [review](https://github.com/apache/systemds/pulls),
let's make sure the changes are consistent with the guidelines and coding style.

#### General Guidelines and Philosophy for contribution

*   Inclusion of unit tests when contributing new features, will help
    1. prove that the code works correctly, and
    2. guard against future breaking changes.
*   Formatting changes can be handled in a separate PR.
    Example [`bf4ba16b`](https://github.com/apache/systemds/commit/bf4ba16b9aaa9afee20a3f1c03b0ff49c5346a9d)
*   New features (e.g., a new cutting edge machine learning algorithm) typically will
    live in [scripts/staging](./scripts/staging) or its equivalent folder for specific
    feature to get some airtime and sufficient testing before a decision is made regarding
    whether they are to migrated to the top-level.
*   When a new contribution is made to SystemDS, the maintenance burden is (by default)
    transferred to the SystemDS team. The benefit of the contribution is to be compared
    against the cost of maintaining the feature.

#### Code Style

Before contributing a pull request, we highly suggest applying a code formatter to the written code.

We have provided at profile for java located in [Codestyle File ./docs/CodeStyle.eclipse.xml](dev/CodeStyle_eclipse.xml). This can be loaded in most editors e.g.:

- [Eclipse](https://stackoverflow.com/questions/10432538/eclipse-import-conf-xml-files#10433986)
- [IntelliJ](https://imagej.net/Eclipse_code_style_profiles_and_IntelliJ)
- [Visual Studio Code](https://stackoverflow.com/questions/46030629/need-to-import-eclipse-java-formatter-profile-in-visual-studio-code)

#### License

Including a license at the top of new files helps validate the consistency of license.

Examples:

- [C/C++/cmake/cuda](./src/main/cpp/libmatrixdnn.h#L1-L18)
- [Python](./src/main/python/create_python_dist.py#L1-L21)
- [Java](./src/main/java/org/apache/sysds/api/ConfigurableAPI.java#L1-L18)
- [Bash](./src/main/bash/sparkDML2.sh#L2-L21)
- [XML/HTML](./src/assembly/bin.xml#L2-L19)
- [dml/R](./scripts/algorithms/ALS-CG.dml#L1-L20)
- [Makefile/.proto](./src/main/cpp/kernels/Makefile#L1-L18)
- Markdown - refer to the top of this file!


___

Thanks again for taking your time to help improve SystemDS! :+1:
