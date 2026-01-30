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

# Data Integration and Large-Scale Analysis

***Student Project SYSTEMDS-3878*** \
***Improve Docker Security***

>This task is to improve the security of the Docker images we provide. Currently we get an 'F' evaluation on DockerHub, and we would like instead to provide secure images.

In the [appendix](#detailed-example-orgapachezookeeperzookeeper), you can see a full example on how we analysed the identified vulnerabilities.

- [Data Integration and Large-Scale Analysis](#data-integration-and-large-scale-analysis)
  - [Changes](#changes)
    - [`pom.xml`](#pomxml)
    - [Critical CVEs `io.netty/netty@3.10.6.Final`](#critical-cves-ionettynetty3106final)
    - [`sysds.Dockerfile`](#sysdsdockerfile)
  - [Changelog](#changelog)
  - [Appendix](#appendix)
    - [Detailed example: `org.apache.zookeeper/zookeeper`](#detailed-example-orgapachezookeeperzookeeper)

## Changes

### `pom.xml`

By calling this url, you can view the changes on the pom.xml between the start and the end of the SYSDS-3878 student project: \
[https://github.com/qschnee/systemds/compare/05b2298..3d563ff#diff-pom.xml](https://github.com/qschnee/systemds/compare/05b2298..3d563ff#diff-9c5fb3d1b7e3b0f54bc5c4182965c4fe1f9023d449017cece3005d3f90e8e4d8) \
This url compares the two commits and scrolls the webpage directly to the `pom.xml` changes. Note that the real link does not use the filename, but a hash.

### Critical CVEs `io.netty/netty@3.10.6.Final`

This is a transitive dependency from `hadoop-hdfs 3.3.6`. Version `3.4.2` does not use the `netty` dependency anymore. The update has been made in the `pom.xml` property: \
`<hadoop-hdfs.version>3.4.2</hadoop-hdfs.version>`

The only change necessary in the code is in the import of the netty package. What was
```java
import org.jboss.netty.handler.codec.compression.CompressionException;
```
became
```java
import io.netty.handler.codec.compression.CompressionException;
```

#### Usage

At the time of writing, the only usage of this import is in [systemds/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupDDC.java](systemds/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupDDC.java):
```java
// import org.jboss.netty.handler.codec.compression.CompressionException;
import io.netty.handler.codec.compression.CompressionException;
// CVE-2019-20444: org.jboss.netty replaced by io.netty in hadoop-hdfs 3.4.2
```

### `sysds.Dockerfile`

```Dockerfile
RUN apk add --no-cache bash \
    snappy \
    lz4 \
    zlib \
    # added apks update to solve openssl and busybox CVEs
    && apk update \
    && apk upgrade openssl busybox busybox-binsh ssl_client libcrypto3 libssl3
```

## Changelog

#### Dec 4, 2025
- Create documentation to log work and document changes

#### Jan 6, 2026
- `org.apache.zookeeper/zookeeper@3.6.3` -> 3.8.3 (CVE-2023-44981)
- Modified `sysds.Dockerfile` to build locally from filesystem instead of pulling latest (which doesn't contain the changes yet)

#### Jan 11, 2026
- kerby → 2.0.3 (CVE-2023-25613)
- avro → 1.11.4 (CVE-2024-47561)

#### Jan 14, 2026
- `org.apache.zookeeper/zookeeper@3.8.3` -> 3.9.4 (CVE-2024-23944)
- explicit `io.nety/netty-handler@4.1.118.Final` (CVE-2023-3678, CVE-2025-11226, CVE-2024-12798, CVE-2024-12801)

#### Jan 15, 2026
- json-smart → 2.4.9 (CVE-2023-1370)
- jettison → 1.5.4 (CVE-2023-1436)
- snappy-java → 1.1.10.4 (CVE-2023-43642)
- nimbus-jose-jwt → 9.37.4 (CVE-2023-52428)
- logback → 1.2.13 (CVE-2023-6378)
- aircompressor → 0.27 (CVE-2024-36114)
- commons-io → 2.14.0 (CVE-2024-47554)
- protobuf-java → 3.25.5 (CVE-2024-7254)
- jackson-core → 2.15.0 (CVE-2025-52999)

#### Jan 16, 2026
- netty → 4.1.124.Final (CVE-2023-44487, CVE-2025-55163)
- xnio-api → 3.8.14.Final (CVE-2023-5685)
- dnsjava → 3.6.0 (CVE-2024-25638)
- jetty → 9.4.57.v20241219 (CVE-2024-6763)
- commons-beanutils → 1.11.0 (CVE-2025-48734)

#### Jan 21, 2026
- `org.eclipse.jetty/jetty-http@9.4.57.v20241219` -> 12.0.12 (CVE-2024-6763)
- explicit `org.apache.commons/commons-compress@1.26.0` (CVE-2024-26308, CVE-2024-25710, CVE-2023-42503)
- explicit `org.apache.commons/commons-configuration2@2.10.1` (CVE-2024-29133, CVE-2024-29131)
- explicit `org.apache.hadoop.thirdparty/hadoop-shaded-guava@1.5.0`
- explicit `com.google.guava/guava@33.5.0-jre`

  --> (CVE-2023-2976, CVE-2020-8908)

#### Jan 22, 2026
- `io.netty/netty-codec-http@4.1.124.Final` -> 4.1.129.Final
- `io.netty/netty-codec-smtp@4.1.124.Final` -> 4.1.129.Final
- `io.netty/netty-codec@4.1.124.Final` -> 4.1.129.Final

--> (CVE-2025-67735, CVE-2025-58056, CVE-2025-58057, CVE-2025-59419)
- `org.apache.logging.log4j/log4j-core@2.22.1` -> 2.25.3 (CVE-2025-68161)

#### Jan 23, 2026
- explicit `org.apache.commons/commons-lang3@3.18.0` (CVE-2025-48924)
- explicit `org.apache.spark/spark-network-common_2.12@3.5.2` (CVE-2025-55039)
- `org.apache.hadoop/hadoop-common@3.3.6` -> 3.4.2 (CVE-2024-23454) \
  Version 3.4.0 is enough for CVE-2024-23454, but generates two other CVEs which are solved with upgraded version 3.4.2
- `org.apache.hadoop/hadoop-hdfs@3.3.6` -> 3.4.2 (CVE-2019-20444) \
  Import changes: `org.jboss.netty.*` → `io.netty.*`

#### Jan 25, 2026
- busybox → 1.36.1-r31 (CVE-2024-58251, CVE-2025-46394)
- openssl → 3.3.5-r0 (CVE-2025-9230, CVE-2025-9231, CVE-2025-9232)


#### Jan 29, 2026
- Created `UNFIXED_VULNERABILITIES.md` documenting 16 remaining CVEs

## Appendix

### Detailed example: `org.apache.zookeeper/zookeeper`

Here is the detail of how to detect and solve vulnerabilities once identified by zookeeper.

CVE-2023-44981 Authorization Bypass Through User-Controlled Key
- Analyse the vulnerable package

  The current version is `3.6.3`. Scout recommends upgrading the package to `3.7.2`, `3.8.3`, or `3.9.1`. \
  [Releases](https://zookeeper.apache.org/releases.html): `3.7.2` already reached its EoL.

  Apache zookeeper is not directly imported in the project. The vulnerability is raised transitively by another dependecy. \
  `mvn dependency:tree` shows all implicit dependencies (see the [appendix](../README.md#output-of-mvn-dependencytree-before-any-change) for the output). \
  The responsible dependency seems to be the apache spark-core package.

  ```bash
  [INFO] org.apache.systemds:systemds:jar:3.4.0-SNAPSHOT
  [INFO] +- org.apache.spark:spark-core_2.12:jar:3.5.0:compile
  [INFO] |  +- org.apache.zookeeper:zookeeper:jar:3.6.3:compile
  ```

  From the documentation [Spark Project Core](https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.13/4.0.1): Version `3.5.x` uses zookeeper `3.6.3`. \
  The version `4.0.x` uses zookeeper `3.9.3` which still has a known vulnerability.

  Spark-core `4.1.0` uses the latest version `3.9.4` of zookeeper.

  By inspecting the jars, it has been found out that zookeeper is used by `hadoop-common` as well.
- Ideas to solve Vulnerabilities
  - Upgrade `spark-core`
    - `3.4.x` doesn't use scala `2.12` anymore \
      Update other dependencies (`spark-sql` etc) for the whole project to use scala `2.13`. \
      scala `2.13` is not backwards compatible with `2.12`.
    - No direct backwards compatibility between `4.0` and `3.5` \
      [core migration guide](https://spark.apache.org/docs/latest/core-migration-guide.html#upgrading-from-core-35-to-40) \
      This upgrade would require significant changes on the whole project.
  - Make `spark-core` use zookeeper `3.8.3+`
    - as `spark-core` is a package  using `zookeeper`, explicitely declaring another version of the zookeeper dependency will override spark's dependecy. As zookeeper is backwards compatible, this should not cause problems for the app
    - `hadoop-common 3.3.6` also uses `zookeeper 3.6.3`.

#### Solution

As `org.apache.zookeeper/zookeeper@3.8.3` also has transitive vulnerabilities, it has been decided to upgrade to `3.9.4` which only has one known transitive vulnerability. \
Zookeeper `3.9.4` will be used to solve all zookeeper-related vulnerabilities. `netty-handler` is also explicitly imported to solve a vulnerability related to the implicit version.

**Changes** 
- `pom.xml`
  - explicit dependency \
    `zookeeper 3.9.4` \
    The vulnerability is solved by upgrading the version of `zookeeper`
    ```xml
    <dependency>
      <groupId>io.netty</groupId>
      <artifactId>netty-handler</artifactId>
      <version>${netty.version}</version>
    </dependency>
    <!-- 
    Explicit declaration to use 3.9.4 instead of implicit use of 3.6.3 by spark-core and hadoop-common.
    Solves critical vulnerability CVE-2023-44981 in the docker image.
    -->
    <dependency>
      <groupId>org.apache.zookeeper</groupId>
      <artifactId>zookeeper</artifactId>
      <version>3.9.4</version>
      <exclusions>
        <exclusion>
          <groupId>io.netty</groupId>
          <artifactId>netty-handler</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
    ```
  - exclude `zookeeper` from
    - `spark-core`
    - `hadoop-common`

    Excluding `zookeeeper` from the dependencies prevents them from using a bundled older version and creating conflicts.
    ```xml
    <exclusion>
      <groupId>org.apache.zookeeper</groupId>
      <artifactId>zookeeper</artifactId>
    </exclusion>
    ```

**Verification**
- `docker scout` \
  The critical vulnerability in `zookeeper` is not shown anymore
- `mvn dependency:tree` \
  shows a more recent version of zookeeper

  No packet imported by zookeeper generates CVES.
- building the project works \
  `mvn clean package -P distribution`
