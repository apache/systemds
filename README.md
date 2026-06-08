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

# Apache SystemDS

**Overview:** Apache SystemDS is an open-source machine learning (ML) system for the end-to-end 
data science lifecycle from data preparation and cleaning, over efficient ML model training, 
to debugging and serving. ML algorithms or pipelines are specified in a high-level language 
with R-like syntax or related Python and Java APIs (with many builtin primitives), and the 
system automatically generates hybrid runtime plans of local, in-memory operations and distributed
operations on Apache Spark. Additional backends exist for GPUs and federated learning. 

Resource | Links
---------|------
**Quick Start** | [Install, Quick Start and Hello World](https://apache.github.io/systemds/site/install.html)
**Documentation:** | [SystemDS Documentation](https://apache.github.io/systemds/)
**Python Documentation** | [Python SystemDS Documentation](https://apache.github.io/systemds/api/python/index.html)
**Issue Tracker** | [Jira Dashboard](https://issues.apache.org/jira/secure/Dashboard.jspa?selectPageId=12335852)


**Status and Build:** SystemDS is renamed from SystemML which is an **Apache Top Level Project**.
To build from source visit [SystemDS Install from source](https://apache.github.io/systemds/site/install.html)

[![Build](https://github.com/apache/systemds/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/build.yml)
[![Documentation](https://github.com/apache/systemds/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/documentation.yml)
[![LicenseCheck](https://github.com/apache/systemds/actions/workflows/license.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/license.yml)
[![Java Tests](https://github.com/apache/systemds/actions/workflows/javaTests.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/javaTests.yml)
[![codecov](https://codecov.io/gh/apache/systemds/graph/badge.svg?token=4YfvX8s6Dz)](https://codecov.io/gh/apache/systemds)
[![Python Test](https://github.com/apache/systemds/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/python.yml)
[![Total PyPI downloads](https://static.pepy.tech/personalized-badge/systemds?units=abbreviation&period=total&left_color=grey&right_color=blue&left_text=Total%20PyPI%20Downloads)](https://pepy.tech/project/systemds)
[![Monthly PyPI downloads](https://static.pepy.tech/personalized-badge/systemds?units=abbreviation&left_color=grey&right_color=blue&left_text=Monthly%20PyPI%20Downloads)](https://pepy.tech/project/systemds)

## Term Project Plan: OOC Operator Extensions

This section tracks the seven-week term project for extending Apache SystemDS Out-of-Core (OOC)
execution support. The implementation work is planned across six active development weeks, with the
final week reserved for review feedback, CI fixes, benchmark polishing, and final submission.

The project has two operator-extension goals:

1. Add OOC support for covariance operations: `cov(A, B)` and `cov(A, B, W)`.
2. Extend `TSMMOOCInstruction`, the specialized transposed self matrix multiplication instruction,
   so it supports arbitrary output dimensions instead of only single-output-tile cases.

### Team Roles

| Role | Main Responsibility |
| --- | --- |
| Person A | Covariance OOC implementation lead |
| Person B | TSMM OOC implementation lead |
| Person C | Testing, benchmarking, integration, and PR quality lead |

All team members should review each other's code weekly. Each major task should be developed in
small commits with tests.

### Relevant Source Files

#### Covariance

| Purpose | File |
| --- | --- |
| CP covariance reference | `src/main/java/org/apache/sysds/runtime/instructions/cp/CovarianceCPInstruction.java` |
| Spark covariance reference | `src/main/java/org/apache/sysds/runtime/instructions/spark/CovarianceSPInstruction.java` |
| OOC central moment reference | `src/main/java/org/apache/sysds/runtime/instructions/ooc/CentralMomentOOCInstruction.java` |
| OOC parser | `src/main/java/org/apache/sysds/runtime/instructions/OOCInstructionParser.java` |
| OOC instruction type enum | `src/main/java/org/apache/sysds/runtime/instructions/ooc/OOCInstruction.java` |

#### TSMM

| Purpose | File |
| --- | --- |
| Current OOC TSMM implementation | `src/main/java/org/apache/sysds/runtime/instructions/ooc/TSMMOOCInstruction.java` |
| OOC matrix multiplication reference | `src/main/java/org/apache/sysds/runtime/instructions/ooc/MMultOOCInstruction.java` |
| Spark multi-block TSMM reference | `src/main/java/org/apache/sysds/runtime/instructions/spark/Tsmm2SPInstruction.java` |
| Current OOC TSMM test | `src/test/java/org/apache/sysds/test/functions/ooc/TransposeSelfMMTest.java` |
| Current OOC TSMM DML script | `src/test/scripts/functions/ooc/TSMM.dml` |

### Weekly Deliverables by Person

| Week | Person A: Covariance Lead | Person B: TSMM Lead | Person C: Testing, Benchmarking, and Integration Lead |
| --- | --- | --- | --- |
| Week 1 | Understand CP/Spark covariance references; identify `CmCovObject`, `COVOperator`, and `MatrixBlock.covOperations(...)`; create failing OOC tests for `cov(A, B)`. | Understand current `TSMMOOCInstruction`; identify the single-output-tile limitation; create failing tests for multi-tile `t(X) %*% X`. | Verify local setup, Maven, Java, and IDE style; define benchmark dimensions, sparsities, block sizes, and CP/OOC/Spark comparison plan. |
| Week 2 | Create `CovarianceOOCInstruction.java`; implement unweighted `cov(A, B)`; wire covariance into `OOCInstructionParser.java`. | Design TSMM output tile indexing; identify diagonal and off-diagonal tile generation; prototype helper logic for output indexes. | Add OOC covariance DML script; run CP vs OOC correctness checks for unweighted covariance; document first test results. |
| Week 3 | Implement weighted `cov(A, B, W)`; join `A`, `B`, and `W` streams; add dimension and block-size validation. | Replace the single-block TSMM aggregation with streamed output tiles; generate diagonal and off-diagonal partial outputs. | Add weighted covariance tests and DML script; start benchmark runner with JVM warm-up support. |
| Week 4 | Stabilize covariance; add dense, sparse, weighted, unweighted, invalid-dimension, and error-handling tests; verify OOC heavy hitters. | Complete multi-block TSMM for `t(X) %*% X`; verify output matrix characteristics and symmetric off-diagonal tiles; add `X %*% t(X)` support if feasible. | Extend `TransposeSelfMMTest`; compare TSMM OOC output against CP output; confirm OOC execution via heavy hitters. |
| Week 5 | Run covariance benchmarks for dense/sparse and weighted/unweighted cases; compare CP vs OOC and Spark where practical. | Run TSMM benchmarks for single-tile and multi-tile outputs; test dense/sparse inputs and multiple block sizes. | Collect benchmark tables; record dimensions, sparsity, block size, execution mode, warm-up count, measured runtime, and limitations. |
| Week 6 | Finalize covariance code; remove debug code; run focused covariance tests; review code style. | Finalize TSMM code; simplify helper methods; run focused TSMM tests; review memory behavior. | Run integration tests; prepare PR description with summary, tests, benchmark methodology, benchmark results, and known limitations. |

### Expected Deliverables

- OOC support for unweighted covariance: `cov(A, B)`.
- OOC support for weighted covariance: `cov(A, B, W)`.
- Extended OOC TSMM support for arbitrary output dimensions, not only single-tile outputs.
- Correctness tests for dense and sparse inputs.
- Error handling tests for invalid dimensions or unsupported cases.
- Benchmarks comparing OOC with CP, and Spark where practical.
- Benchmark results with warm-up runs.
- A clean Apache SystemDS pull request before the deadline.

### Weekly Review Checklist

| Week | Required Review Output |
| --- | --- |
| Week 1 | Failing tests exist and fail for the expected reason. |
| Week 2 | First covariance implementation is wired into OOC parsing. |
| Week 3 | Weighted covariance and TSMM prototype are both reviewable. |
| Week 4 | Correctness tests pass for the main covariance and TSMM cases. |
| Week 5 | Benchmark data is collected in a reproducible format. |
| Week 6 | PR is clean, documented, and ready for review before the deadline. |

### Suggested Test Commands

```bash
mvn -Dtest=org.apache.sysds.test.functions.ooc.CovarianceTest test
mvn -Dtest=org.apache.sysds.test.functions.ooc.CovarianceWeightsTest test
mvn -Dtest=org.apache.sysds.test.functions.ooc.TransposeSelfMMTest test
```

### Success Criteria

- `cov(A, B)` works in OOC mode.
- `cov(A, B, W)` works in OOC mode.
- TSMM works when the output has more than one tile.
- Correctness tests compare OOC output against CP output.
- Tests cover dense and sparse inputs.
- Benchmarks include warm-up runs.
- The PR follows Apache SystemDS style conventions.

### Main Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| TSMM multi-block output is more complex than expected | Start TSMM failing tests in Week 1 and prototype in Week 2. |
| OOC stream joins cause memory pressure | Reuse existing OOC primitives and avoid full materialization. |
| Benchmarks are noisy | Use warm-up runs and repeat measurements. |
| PR becomes too large | Keep covariance and TSMM in separate commits, possibly separate PRs if advised. |
| Late reviewer feedback | Open the PR before the deadline, even if benchmark polishing is still ongoing. |
