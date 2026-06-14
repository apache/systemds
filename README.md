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


### Project Summary

This term project extends Apache SystemDS Out-of-Core (OOC) execution support for two operators:

1. `cov(A, B)` and `cov(A, B, W)` covariance operations.
2. `TSMMOOCInstruction`, the specialized transposed self matrix multiplication instruction used for both `t(X) %*% X` and `X %*% t(X)`.

The project originally had a six-week implementation plan, but the schedule is now compressed to four active weeks. The revised scope emphasizes test-first development, includes both TSMM directions, removes Spark benchmarking from scope, includes performance tests in the PR, and opens the draft PR early enough for external feedback.

### Team Roles

| Role | Main Responsibility |
| --- | --- |
| Person A | Covariance OOC implementation lead |
| Person B | TSMM OOC implementation lead |
| Person C | Testing, benchmarking, integration, and PR coordination lead |

All team members should review each other's code weekly. Each major task should begin with a failing correctness test and should be developed in small commits.

### Updated Scope

| Topic | Plan Update |
| --- | --- |
| Overall plan | Keep the three-person split and compress the schedule to four weeks. |
| Development workflow | Start with failing tests, compare OOC against CP, then implement until tests pass. |
| TSMM direction | Both `t(X) %*% X` and `X %*% t(X)` are required. Right TSMM is no longer optional. |
| Spark benchmarking | Remove Spark benchmarking from scope. Focus on CP vs OOC. |
| Performance tests | Include performance test code in the PR, not only benchmark numbers. |
| PR timing | Open a draft PR early enough for review; target end of Week 3. |
| PR description | Include tests run, benchmark setup, benchmark numbers, and known limitations. |

### Four-Week Deliverables by Person

| Week | Person A: Covariance Lead | Person B: TSMM Lead | Person C: Testing, Benchmarking, and PR Coordination |
| --- | --- | --- | --- |
| Week 1 | Write failing OOC tests for `cov(A, B)`; study CP/Spark covariance and OOC central moment references; start `CovarianceOOCInstruction.java`; wire initial parser support. | Write failing multi-tile tests for both `t(X) %*% X` and `X %*% t(X)`; study `TSMMOOCInstruction`, `MMultOOCInstruction`, and `Tsmm2SPInstruction`; document LEFT/RIGHT output indexing. | Verify setup and code style; add first OOC covariance DML script; define CP-vs-OOC benchmark matrix sizes, sparsities, and block sizes; verify all new tests fail for the expected reason. |
| Week 2 | Complete unweighted covariance; implement weighted `cov(A, B, W)`; add stream joins, `CmCovObject` reduction, dimension checks, and block-size validation. | Remove the single-output-block limitation; produce diagonal and off-diagonal output tiles; implement LEFT and RIGHT TSMM indexing prototype. | Add weighted covariance DML/test files; compare covariance OOC against CP; create performance test skeleton in the existing Java performance test area. |
| Week 3 | Stabilize covariance; add dense/sparse, weighted/unweighted, and error-handling tests; verify covariance OOC heavy hitters; review TSMM implementation. | Complete LEFT and RIGHT multi-block TSMM; verify output metadata and symmetric off-diagonal tiles; review covariance implementation. | Extend `TransposeSelfMMTest`; run focused correctness tests; implement CP-vs-OOC performance tests; collect preliminary numbers; open draft PR and request early review. |
| Week 4 | Address covariance feedback; finalize tests; remove debug code; check comments and style. | Address TSMM feedback; finalize helper methods, memory behavior, and output metadata handling. | Run final correctness tests and warm-up benchmarks; update PR description with final tables, tests run, limitations, and final submission notes. |

### Relevant Source Files

#### Covariance

| Purpose | File |
| --- | --- |
| CP covariance reference | `src/main/java/org/apache/sysds/runtime/instructions/cp/CovarianceCPInstruction.java` |
| Spark covariance reference for algorithmic behavior only | `src/main/java/org/apache/sysds/runtime/instructions/spark/CovarianceSPInstruction.java` |
| OOC central moment reference | `src/main/java/org/apache/sysds/runtime/instructions/ooc/CentralMomentOOCInstruction.java` |
| OOC parser | `src/main/java/org/apache/sysds/runtime/instructions/OOCInstructionParser.java` |
| OOC instruction type enum | `src/main/java/org/apache/sysds/runtime/instructions/ooc/OOCInstruction.java` |

#### TSMM

| Purpose | File |
| --- | --- |
| Current OOC TSMM implementation | `src/main/java/org/apache/sysds/runtime/instructions/ooc/TSMMOOCInstruction.java` |
| OOC matrix multiplication reference | `src/main/java/org/apache/sysds/runtime/instructions/ooc/MMultOOCInstruction.java` |
| Multi-block TSMM reference for algorithmic behavior only | `src/main/java/org/apache/sysds/runtime/instructions/spark/Tsmm2SPInstruction.java` |
| Current OOC TSMM test | `src/test/java/org/apache/sysds/test/functions/ooc/TransposeSelfMMTest.java` |
| Current OOC TSMM DML script | `src/test/scripts/functions/ooc/TSMM.dml` |

### Expected Deliverables

- OOC support for unweighted covariance: `cov(A, B)`.
- OOC support for weighted covariance: `cov(A, B, W)`.
- OOC TSMM support for `t(X) %*% X` with arbitrary output dimensions.
- OOC TSMM support for `X %*% t(X)` with arbitrary output dimensions.
- Correctness tests comparing OOC output against CP output.
- Dense and sparse test cases.
- Error handling tests for invalid dimensions or unsupported cases.
- Performance test code included in the PR.
- Benchmark results included in the PR description.
- A draft PR opened by the end of Week 3 for early review.
- Final cleaned PR before the deadline.

### Four-Week Project Plan

#### Week 1: Test-First Setup and Initial Covariance Implementation

**Goal:** Establish failing correctness tests first, then implement the smallest useful covariance OOC path.

| Person | Weekly Deliverables |
| --- | --- |
| Person A: Covariance Lead | Study `CovarianceCPInstruction.java`, `CovarianceSPInstruction.java`, and `CentralMomentOOCInstruction.java`; create failing OOC tests for `cov(A, B)`; implement initial `CovarianceOOCInstruction.java`; wire covariance into `OOCInstructionParser.java`. |
| Person B: TSMM Lead | Study `TSMMOOCInstruction.java`, `MMultOOCInstruction.java`, and `Tsmm2SPInstruction.java`; create failing tests for multi-tile `t(X) %*% X`; create failing tests for multi-tile `X %*% t(X)`; document output tile indexing for both directions. |
| Person C: Testing, Benchmarking, Integration | Verify Java, Maven, and IDE style setup; add OOC covariance DML script; define CP-vs-OOC benchmark dimensions, sparsities, and block sizes; confirm all new tests fail for expected missing-support reasons. |

##### Week 1 Required Output

- Failing unweighted covariance OOC tests.
- Failing LEFT TSMM multi-tile tests for `t(X) %*% X`.
- Failing RIGHT TSMM multi-tile tests for `X %*% t(X)`.
- Initial covariance parser wiring started or completed.
- Benchmark plan focused only on CP vs OOC.

#### Week 2: Complete Covariance and Build TSMM Multi-Block Prototype

**Goal:** Finish covariance correctness and produce a reviewable TSMM prototype for both directions.

| Person | Weekly Deliverables |
| --- | --- |
| Person A: Covariance Lead | Complete unweighted `cov(A, B)`; implement weighted `cov(A, B, W)`; join `A`, `B`, and `W` streams by block index; reduce `CmCovObject` partials; add dimension and block-size validation. |
| Person B: TSMM Lead | Replace the single-output-block restriction in `TSMMOOCInstruction.java`; generate diagonal partial output tiles; generate off-diagonal partial output tiles; handle both LEFT and RIGHT TSMM indexing. |
| Person C: Testing, Benchmarking, Integration | Add `CovarianceWeights.dml` and `CovarianceWeightsTest.java`; compare covariance OOC results against CP; add initial performance test skeleton under the existing Java performance test area. |

##### Week 2 Required Output

- Dense and sparse unweighted covariance tests passing.
- Dense and sparse weighted covariance tests passing or close to passing.
- TSMM prototype emits multi-tile outputs for both directions.
- Performance test code skeleton exists and is committed.

#### Week 3: TSMM Correctness, Benchmarks, and Early Draft PR

**Goal:** Make both operators reviewable, collect first benchmark numbers, and open the draft PR early enough for feedback.

| Person | Weekly Deliverables |
| --- | --- |
| Person A: Covariance Lead | Stabilize covariance implementation; add error handling tests; verify OOC heavy hitters include covariance; review Person B's TSMM code. |
| Person B: TSMM Lead | Complete multi-block TSMM for `t(X) %*% X`; complete multi-block TSMM for `X %*% t(X)`; verify output matrix characteristics; verify symmetric off-diagonal tiles; review Person A's covariance code. |
| Person C: Testing, Benchmarking, Integration | Extend `TransposeSelfMMTest.java`; run focused correctness tests; implement CP-vs-OOC performance tests; collect preliminary benchmark numbers; open draft PR and request early review. |

##### Week 3 Required Output

- Covariance implementation complete.
- TSMM LEFT and RIGHT correctness tests passing for main dense and sparse cases.
- Initial benchmark results collected.
- Draft PR opened before the end of the week.
- PR description includes tests run, benchmark setup, preliminary numbers, and known limitations.

#### Week 4: Review Feedback, Final Benchmarks, and Submission Cleanup

**Goal:** Use review feedback, finalize benchmarks, clean the PR, and prepare final submission.

| Person | Weekly Deliverables |
| --- | --- |
| Person A: Covariance Lead | Address covariance review comments; finalize covariance tests; remove debug code; check code style and comments. |
| Person B: TSMM Lead | Address TSMM review comments; finalize TSMM tests; simplify helper methods; check memory behavior and output metadata handling. |
| Person C: Testing, Benchmarking, Integration | Run final focused tests; run final benchmark measurements with warm-up runs; update PR description with final benchmark tables; coordinate final PR cleanup. |

##### Week 4 Required Output

- Review feedback addressed where possible.
- Focused correctness tests passing.
- Performance tests included in the PR.
- Final benchmark results included in PR description.
- PR ready for final grading/submission.

### Weekly Review Checklist

| Week | Required Review Output |
| --- | --- |
| Week 1 | Failing tests exist and fail for expected missing OOC support. |
| Week 2 | Covariance is mostly complete; TSMM multi-block prototype is reviewable. |
| Week 3 | Both operators are correctness-testable; draft PR is open for review. |
| Week 4 | Review feedback is addressed; final tests and benchmarks are documented. |

### Suggested Git Workflow

Use one feature branch with separate commits for covariance, TSMM, tests, and benchmarks:

```bash
git switch -c term-ooc-covariance-tsmm
```

Recommended commit structure:

```text
test: add failing OOC covariance tests
feat: add OOC covariance instruction
test: add weighted OOC covariance tests
test: add multi-block OOC TSMM tests
feat: extend OOC TSMM to left and right multi-block outputs
bench: add OOC covariance and TSMM performance tests
docs: add benchmark results to PR notes
```

### Suggested Test Commands

```bash
mvn -Dtest=org.apache.sysds.test.functions.ooc.CovarianceTest test
mvn -Dtest=org.apache.sysds.test.functions.ooc.CovarianceWeightsTest test
mvn -Dtest=org.apache.sysds.test.functions.ooc.TransposeSelfMMTest test
```

Run broader OOC tests if time allows:

```bash
mvn -Dtest=org.apache.sysds.test.functions.ooc.* test
```

### Benchmark Requirements

Benchmarks should compare CP vs OOC only. Spark benchmarking is intentionally removed from scope.

Benchmark variables:

- covariance weighted vs unweighted
- dense vs sparse inputs
- TSMM LEFT: `t(X) %*% X`
- TSMM RIGHT: `X %*% t(X)`
- single-tile output vs multi-tile output
- block sizes such as 500, 1000, and 2000 if feasible
- JVM warm-up runs before measured runs

Benchmark output should include:

- matrix dimensions
- sparsity
- block size
- execution mode
- warm-up count
- measured runtime
- short interpretation of results

### Success Criteria

- `cov(A, B)` works in OOC mode.
- `cov(A, B, W)` works in OOC mode.
- `t(X) %*% X` works in OOC mode when the output has more than one tile.
- `X %*% t(X)` works in OOC mode when the output has more than one tile.
- Correctness tests compare OOC output against CP output.
- Tests cover dense and sparse inputs.
- Benchmarks include warm-up runs.
- Performance test code is included in the PR.
- Benchmark results are included in the PR description.
- Draft PR is opened by the end of Week 3 for feedback.
- Final PR follows Apache SystemDS style conventions.

### Main Risks

| Risk | Mitigation |
| --- | --- |
| Four weeks is tight for two operators | Start with failing tests immediately and open draft PR by Week 3. |
| TSMM LEFT and RIGHT indexing is complex | Treat both directions as required from Week 1, not as an optional add-on. |
| OOC stream joins cause memory pressure | Reuse existing OOC primitives and avoid full materialization. |
| Benchmarks are noisy | Use warm-up runs and repeat measurements. |
| PR becomes too large | Keep covariance, TSMM, tests, and benchmarks in separate commits. |
| Review feedback arrives late | Request early review as soon as the Week 3 draft PR is open. |

### Notes from Meetings

- The covariance operator already exists in CP and Spark, but not in OOC.
- The implementation should be guided by existing CP/Spark covariance code and OOC central moment code.
- Start by writing failing tests and comparing against CP output.
- TSMM OOC currently only handles cases where the output is a single tile.
- TSMM must support both `t(X) %*% X` and `X %*% t(X)`.
- Good TSMM references are `MMultOOCInstruction` and `Tsmm2SPInstruction`.
- Spark benchmarking is not required and has been removed from this plan.
- Performance test code should ideally be included in the PR.
- Benchmark numbers should be documented in the PR description.
- Open the draft PR early, ideally by the end of Week 3, so there is time for one non-grading review before the deadline.
