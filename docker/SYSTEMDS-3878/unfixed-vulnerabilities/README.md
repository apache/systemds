<!-- 
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# 
-->

# Unfixed Vulnerabilities Report

This document explains the remaining vulnerabilities in the `apache/systemds:latest` Docker image that cannot be resolved at this time due to upstream dependencies.

## Overview

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 6 |
| Medium | 8 |
| Low | 1 |
| Unspecified | 1 |

---

## Transitive Maven Dependencies (Spark/Hadoop Ecosystem)

These vulnerabilities originate from dependencies managed by Apache Spark and Hadoop. Upgrading them independently would break compatibility with the Spark runtime.

### 1. protobuf-java 3.7.1

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2024-7254 | High | 8.7 | 3.25.5 |
| CVE-2022-3510 | High | 7.5 | 3.16.3 |
| CVE-2022-3509 | High | 7.5 | 3.16.3 |
| CVE-2021-22569 | High | 7.5 | 3.16.1 |
| CVE-2022-3171 | Medium | 5.7 | 3.16.3 |
| CVE-2021-22570 | Medium | 5.5 | 3.15.0 |

**Reason:** This is a transitive dependency bundled within Apache Spark's runtime JARs. Although SystemDS's `pom.xml` pins `<protobuf.version>3.25.5</protobuf.version>`, the Docker image still contains protobuf 3.7.1 because:

1. **Spark bundles protobuf internally:** Spark 3.5.x includes protobuf classes within its shaded JARs. Maven dependency management cannot override classes already packaged inside Spark's uber-JARs.

2. **Upgrading Spark would fix this:** Spark 4.0+ includes protobuf 3.25.x ([SPARK-49497](https://issues.apache.org/jira/browse/SPARK-49497)), which resolves all these CVEs.

3. **But Spark 4.0 requires Scala 2.13:** Spark 4.0 dropped support for Scala 2.12 entirely. SystemDS currently uses Scala 2.12.18.

4. **Scala 2.12 → 2.13 migration is non-trivial:** This requires recompiling all Scala code, updating the collections API usage, and ensuring all dependencies have Scala 2.13 builds available. This is a significant development effort that SystemDS upstream has not yet completed.

**Dependency chain preventing the fix:**
```
protobuf 3.25.5 fix
    └── requires Spark 4.0+
            └── requires Scala 2.13
                    └── requires SystemDS codebase migration (not yet done upstream)
```

**References:**
- [SPARK-49497: Upgrade protobuf-java to 3.25.4](https://issues.apache.org/jira/browse/SPARK-49497)
- [Spark 4.0 Release Notes - Scala 2.12 dropped](https://spark.apache.org/releases/spark-release-4-0-0.html)
- [CVE-2024-7254 Advisory](https://advisories.gitlab.com/pkg/maven/com.google.protobuf/protobuf-java/CVE-2024-7254)

---

### 2. jetty-server 9.4.52.v20230823

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2024-13009 | High | 7.2 | 9.4.57 |
| CVE-2024-8184 | Medium | 5.9 | 9.4.56 |

**Reason:** Jetty 9.4.x is embedded within Apache Spark 3.x for the Spark UI and REST APIs. Although SystemDS's `pom.xml` pins `<jetty.version>9.4.57.v20241219</jetty.version>`, Spark bundles Jetty internally in its distribution JARs.

1. **Jetty 9.4 is EOL:** Community support ended June 1, 2022; security support ended February 19, 2025.

2. **Spark 4.0 uses Jetty 11+:** Spark 4.0 migrated from Jetty 9.4.56 to 11.0.24, fixing these CVEs.

3. **Same Scala 2.13 blocker:** Upgrading to Spark 4.0 to get the newer Jetty requires Scala 2.13, which SystemDS doesn't yet support.

**Dependency chain preventing the fix:**
```
Jetty 11+ fix
    └── requires Spark 4.0+ (javax → jakarta migration)
            └── requires Scala 2.13
                    └── requires SystemDS codebase migration (not yet done upstream)
```

**References:**
- [Endoflife: Eclipse Jetty](https://endoflife.date/eclipse-jetty)
- [Spark 4.0 Release Notes - Jetty upgraded](https://spark.apache.org/releases/spark-release-4-0-0.html)
- [CVE-2024-13009 Jetty Announcement](https://www.eclipse.org/lists/jetty-announce/msg00197.html)

---

### 3. jetty-servlets 9.4.52.v20230823

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2024-9823 | Medium | 5.3 | 9.4.54 |

**Reason:** Is (same as jetty-server) managed by Apache Spark's dependency tree. The DoSFilter vulnerability affects session tracking and can cause OutOfMemory errors under attack conditions.

**References:**
- [CVE-2024-9823 GitHub Advisory](https://github.com/jetty/jetty.project/security/advisories/GHSA-7hcf-ppf8-5w5h)

---

### 4. jetty-http 9.4.52.v20230823

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2024-6763 | Medium | 6.3 | 12.0.12 |

**Reason:** The `HttpURI` class validation issue is only fully fixed in **Jetty 12.0.12**. This CVE cannot be fixed because:

1. **Spark 3.5.x uses Jetty 9.4.x** - bundled internally, cannot be overridden.

2. **Spark 4.0 uses Jetty 11.0.24** - still does NOT include the fix for CVE-2024-6763.

3. **Fix requires Jetty 12.0.12** - No current Spark version uses Jetty 12.x yet.

**References:**
- [CVE-2024-6763 GitHub Advisory](https://github.com/jetty/jetty.project/security/advisories/GHSA-qh8g-58pp-2wxh)
- [Spark 4.0 Release Notes - Jetty 11](https://spark.apache.org/releases/spark-release-4-0-0.html)
- [Spark 3.5 Release Notes - Jetty 9.4](https://spark.apache.org/releases/spark-release-3-5-0.html)

---

### 5. jackson-core 2.13.4

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2025-52999 | High | 8.7 | 2.15.0 |

**Reason:** The vulnerable jackson-core 2.13.4 is **shaded inside `parquet-jackson-1.13.1.jar`**. The classes cannot be excluded via Maven dependency management.

**Why it cannot be fixed:**
1. The main application correctly uses jackson-core 2.15.0 (the shaded copy is isolated inside Parquet).
2. Upgrading to Parquet 1.14.x (which bundles jackson 2.17.0) was attempted but did not fix the CVE.
3. Upgrading Spark to 4.0 would fix this, but requires Scala 2.13 migration.

**References:**
- [CVE-2025-52999 NVD Entry](https://nvd.nist.gov/vuln/detail/CVE-2025-52999)
- [Jackson-core PR #943](https://github.com/FasterXML/jackson-core/pull/943)
- [Spark 4.0 Release Notes - Parquet 1.15.2](https://spark.apache.org/releases/spark-release-4-0-0.html)

---

### 6. guava 14.0.1

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2018-10237 | Medium | 5.9 | 24.1.1 |
| CVE-2023-2976 | Medium | 5.5 | 32.0.0 |
| CVE-2020-8908 | Low | 3.3 | 32.0.0 |

**Reason:** Guava 14.0.1 is an extremely old version pulled in as a transitive dependency from the Hadoop ecosystem. Hadoop has historically struggled with Guava version conflicts—different components require different versions. The solution implemented in Hadoop 3.3+ is to shade Guava into a separate namespace ([HADOOP-14284](https://issues.apache.org/jira/browse/HADOOP-14284)), but this doesn't help when older unshaded versions are still pulled in transitively.

**References:**
- [HADOOP-14284: Shade Guava everywhere](https://issues.apache.org/jira/browse/HADOOP-14284)
- [HADOOP-17288: Use shaded guava from thirdparty](https://issues.apache.org/jira/browse/HADOOP-17288)
- [HADOOP-16924: Shade & Update guava to 29.0-jre](https://issues.apache.org/jira/browse/HADOOP-16924)

---

## Alpine Linux Base Image Packages

These vulnerabilities are in Alpine Linux system packages. Patched versions must come from Alpine maintainers.

### 7. busybox 1.36.1-r31

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2025-60876 | Medium | 6.5 | **Not Fixed** |

**Reason:** CRLF injection vulnerability in BusyBox wget through version 1.37. Allows attackers to inject control bytes into HTTP request-targets, enabling request line splitting and header injection. No patched version is available in the Alpine Linux 3.20 repository yet.

**References:**
- [CVE-2025-60876 NVD Entry](https://nvd.nist.gov/vuln/detail/CVE-2025-60876)
- [Alpine Linux Security Tracker](https://security.alpinelinux.org/vuln/CVE-2025-60876)
- [Docker Scout CVE-2025-60876](https://scout.docker.com/v/CVE-2025-60876)

---

### 8. lz4 1.9.4-r5

| CVE | Severity | CVSS | Fixed In |
|-----|----------|------|----------|
| CVE-2025-62813 | Unspecified | - | **REJECTED** |

**Reason:** This CVE has been **withdrawn and marked as rejected** by NIST. The CNA determined after investigation that this was not actually a security issue. The originally reported NULL pointer check issue in `LZ4F_createCDict_advanced` was not exploitable. **This can be safely ignored.**

**References:**
- [CVE-2025-62813 NVD Entry (Rejected)](https://nvd.nist.gov/vuln/detail/CVE-2025-62813)

---

*Last updated: January 28, 2026*  
*Source: Docker Scout vulnerability scan (`scan-after-fixes/README.md`)*
