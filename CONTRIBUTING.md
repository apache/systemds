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

# Contribution Guidelines and Standards for `SystemDS`

Thank you for contributing to SystemDS. :smile:

> The following are mostly guidelines, not rules. Use your best judgement, and
> feel free to propose changes to this doc.

Before contributing a pull request for [review](https://github.com/apache/systemds/pulls),
let's make sure the changes are consistent with the guidelines and coding style.

## General Guidelines and Philosophy for contribution

*   Inclusion of unit tests when contributing new features, will help
    1. prove that the code works correctly, and
    2. guard against future breaking changes.
*   Formatting changes can be handled in a separate PR.
    Example [`bf4ba16b`](https://github.com/apache/systemds/commit/bf4ba16b9aaa9afee20a3f1c03b0ff49c5346a9d)
*   New features (e.g., a new cutting edge machine learning algorithm) typically will
    live in [`scripts/staging`](./scripts/staging) or its equivalent folder for specific
    feature to get some airtime and sufficient testing before a decision is made regarding
    whether they are to migrated to the top-level.
*   When a new contribution is made to SystemDS, the maintenance burden is (by default)
    transferred to the SystemDS team. The benefit of the contribution is to be compared
    against the cost of maintaining the feature.

## Code Style

We suggest applying a code formatter to the written code. Generally, this is done automatically.

We have provided at profile for java located in [Codestyle File `./dev/CodeStyle.eclipse.xml`](dev/CodeStyle_eclipse.xml). This can be loaded in most editors e.g.:

- [Eclipse](https://stackoverflow.com/questions/10432538/eclipse-import-conf-xml-files#10433986)
- [IntelliJ](https://imagej.net/Eclipse_code_style_profiles_and_IntelliJ)
- [Visual Studio Code](https://stackoverflow.com/questions/46030629/need-to-import-eclipse-java-formatter-profile-in-visual-studio-code)

## Commit Style

SystemDS project has linear history throughout. Rebase cleanly and verify that the commit-SHA's
of the Apache Repo are not altered.
A general guideline is never to change a commit inside the systemds official repo, which is
the standard practice. If you have accidentally committed a functionality or tag, let others know.
And create a new commit to revert the functionality of the previous commit but **do not force push**.


### Tags

The tags can be used in combination to one another. These are the only tags available.

* `[MINOR]`: Small changesets with additional functionality

  > Examples:
  >
  > This commit makes small software updates with refactoring
  > 
  > [`030fdab3`](https://github.com/apache/systemds/commit/030fdab3ebe6dedc3b4bb860e0ec5acfd9c38e5d) - `[MINOR] Added package to R dependency; updated Docker test image`
  >
  > This commit cleans up the redundant code for simplification
  > 
  > [`f4fa5650`](https://github.com/apache/systemds/commit/f4fa565013de13270df05dd37610382ca80f7354) - `[MINOR][SYSTEMDS-43] Cleanup scale builtin function (readability)`
  >

* `[SYSTEMDS-#]`: A changeset which will a specific purpose such as a bug, Improvement, 
   New feature, etc. The tag value is found from [SystemDS jira issue tracker](https://issues.apache.org/jira/projects/SYSTEMDS/issues).
   
* `[DOC]` also `[DOCS]`: Changes to the documentation
  
* `[HOTFIX]`: Introduces changes into the already released versions.
    
    > Example:
    > 
    > This commit fixes the corrupted language path
    > 
    > [`87bc3584`](https://github.com/apache/systemds/commit/87bc3584db2148cf78b2d46418639e88ca27ec64) - `[HOTFIX] Fix validation of scalar-scalar binary min/max operations`
    >

> Protip:
> Addressing multiple jira issues in a single commit, `[SYSTEMDS-123,SYSTEMDS-124]` or `[SYSTEMDS-123][SYSTEMDS-124]`

### Commit description

A commit or PR description is a public record of **what** change is being made and **why**.

#### Structure of the description

##### First Line

1. A summary of what the changeset.
2. A complete sentence, crafted as though it was an order.
    - an imperative sentence
    - Writing the rest of the description as an imperative is optional.
3. Follow by an empty line.

##### Body

It consists of the following.

1. A brief description of the problem solved.
2. Why this is the best approach?.
3. Shortcomings to the approach, if any (important!).

Additional info

4. background information
   - bug/issue/jira numbers
   - benchmark/test results
   - links to design documents
   - code attributions
5. Include enough context for
   - reviewers
   - future readers to understand the Changes.
6. Add PR number, like `Closes #1000`.

The following is a commit description with all the points mentioned.

[`1abe9cb`](https://github.com/apache/systemds/commit/1abe9cb79d8001992f1c79ba5e638e6b423a1382)

Commit message:
```txt
[SYSTEMDS-418] Performance improvements lineage reuse probing/spilling

This patch makes some minor performance improvements to the lineage
reuse probing and cache put operations. Specifically, we now avoid
unnecessary lineage hashing and comparisons by using lists instead of
hash maps, move the time computations into the reuse path (to not affect
the code path without lineage reuse), avoid unnecessary branching, and
materialize the score of cache entries to avoid repeated computation
for the log N comparisons per add/remove/constaints operation.
For 100K iterations and ~40 ops per iteration, lineage tracing w/ reuse
improved from 41.9s to 38.8s (pure lineage tracing: 27.9s).
```

#### Good commit description

The following are some of the types of code changes with examples.

##### Functionality change

[`1101533`](https://github.com/apache/systemds/commit/1101533fd1b2be4e475a18052dbb4bc930bb05d9)

Commit message:
```txt
[SYSTEMDS-2603] New hybrid approach for lineage deduplication

This patch makes a major refactoring of the lineage deduplication
framework. This changes the design of tracing all the
distinct paths in a loop-body before the first iteration, to trace
during execution. The number of distinct paths grows exponentially
with the number of control flow statements. Tracing all the paths
in advance can be a huge waste and overhead.
We now trace an iteration during execution. We count the number of
distinct paths before the iterations start, and we stop tracing
once all the paths are traced. Tracing during execution fits
very well with our multi-level reuse infrastructure.
Refer to JIRA for detailed discussions.
```


##### Refactoring

> Refactoring is a series of behaviour preserving changes with restructuring to
> existing code body. Refactoring does not alter the external behaviour or the
> output of a function, but the internal changes to the function to keep the code
> more organized, readable or to accomodate a more complex functionality.


[`e581b5a`](https://github.com/apache/systemds/commit/e581b5a6248b56a70e18ffe6ba699e8142a2d679)

Commit message:
```txt
[SYSTEMDS-2575] Fix eval function calls (incorrect pinning of inputs)

This patch fixes an issue of indirect eval function calls where wrong
input variable names led to missing pinning of inputs and thus too eager
cleanup of these variables (which causes crashes if the inputs are used
in other operations of the eval call).
The fix is simple. We avoid such inconsistent construction and
invocation of fcall instructions by using a narrower interface and
constructing the materialized names internally in the fcall.
```

##### Small Changeset still needs some context.

[`7af2ae0`](https://github.com/apache/systemds/commit/7af2ae04f28ddcb36158719a25a7fa34b22d3266)

Commit message:
```txt
[MINOR] Update docker images organization

Changes the docker images to use the docker organization systemds
add install dependency for R dbScan
Change the tests to use the new organizations docker images

Closes #1008
```

> Protip: to reference other commits use first 7 letters of the commit SHA-1.
> eg. `1b81d8c` for referencing `1b81d8cb19d8da6d865b7fca5a095dd5fec8d209`

## License

Including a license at the top of new files helps validate the consistency of license.

Examples:

- [C/C++/cmake/cuda](./src/main/cpp/libmatrixdnn.h#L1-L18)
- [Python](./src/main/python/create_python_dist.py#L1-L21)
- [Java](./src/main/java/org/apache/sysds/api/ConfigurableAPI.java#L1-L18)
- [Bash](./src/main/bash/sparkDML2.sh#L2-L21)
- [XML/HTML](./src/assembly/bin.xml#L2-L19)
- [dml/R](./scripts/algorithms/ALS-CG.dml#L1-L20)
- [Makefile/.proto](./src/main/python/docs/Makefile#L1-L20)
- Markdown - refer to the top of this file in raw format.


___

Thanks again for taking your time to help improve SystemDS! :+1:
