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

## `docker scout cves` identified vulnerabilities

### org.apache.zookeeper/zookeeper

- CVE-2023-44981 Authorization Bypass Through User-Controlled Key
  - Identify the correct package

    Scout recommends upgrading the package to `3.7.2`, `3.8.3`, or `3.9.1`. \
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
