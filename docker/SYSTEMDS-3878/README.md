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
## Overview

This README presents the project 3878 and serves as index for the documentation.

The goal of SYSTEMDS-3878 as presented in [jira](https://issues.apache.org/jira/browse/SYSTEMDS-3878) is to "Improve Docker Security":
>This task is to improve the security of the Docker images we provide. Currently we get an 'F' evaluation on DockerHub, and we would like instead to provide secure images.
https://hub.docker.com/repository/docker/apache/systemds/general

<!-- TODO Table of contents -->

## Working on apache/systemds

### Create a fork

1. Fork systemds github on own account
2. Invite project member
1. Create SSH-key locally on computer
1. Add public key to github account
1. Clone project on computer (git clone git@\<link>.git)
1. Get a local version of the systemds project on the computer

### Commiting

Extract from [CONTRIBUTING.md](https://github.com/qschnee/systemds/blob/main/CONTRIBUTING.md).

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

#### Example for this project:

```text
[MINOR][SYSTEMDS-3878] Fix <file>.Dockerfile vulnerability (<bug>)

This commit fixes the following vulnerability identified by docker scout cves: <...>. The following changes have been made on the Dockerfile: <...>.
<Sources/justifications>
<Shortcomings>
<new CVE if not fully resolved>
```

## Docker scout cves

We used the tool docker scout cves to identify and solve vulenrabilities in the systemds image build from the Dockerfile.

From the man page:
>The docker scout cves command analyzes a software artifact for vulnerabilities. \
If no image is specified, the most recently built image is used. \
[...] \
The tool analyzes the provided software artifact, and generates a vulnerability report.

### Usage

``` sh
cd path/to/systemds
./docker/build.sh
docker scout cves --details --format markdown -o docker/scout_results/sysds_outputX.md --epss apache/systemds
```

To identify the location of tricky packages, using the json output with `--format sarif` helps by showing the path to the vulnerable package (which is not shown in the "human-readable" markdown output).

### Command details

#### Local analysis

To use docker scout and see if local changes succesfully solved a vulnerability, it is necessary to modify `sysds.Dockerfile`. \
`systemds` is cloned from github instead of using the local systemds project. This change should be reverted and rever merged to the master branch.
```Dockerfile
# Build the system
# RUN git clone --depth 1 https://github.com/apache/systemds.git systemds && \
# 	cd /usr/src/systemds/ && \
# 	mvn --no-transfer-progress clean package -P distribution

# Copy the local SystemDS source into the image
COPY . /usr/src/systemds
# Build SystemDS
RUN cd /usr/src/systemds && \
   mvn --no-transfer-progress clean package -P distribution
```

#### Options

<https://docs.docker.com/reference/cli/docker/scout/cves/>

1. build systemds project \
   `./docker/build.sh`
   - builds the image with the tag: `apache/systemds:latest`
   - default runs: `docker image build -f docker/sysds.Dockerfile -t apache/systemds:latest .`
   - modify `docker/build.sh` to change the selected `Dockerfile` for the build
2. scout \
   `docker scout cves --details --format markdown -o <docker_subdirecory>/scout_results/<file>_output0.md --epss apache/systemds` (from systemds directory)
   - `--details`: verbose
   - `--format markdown`: output in markdown format
   - `-o <path_to_file>`: write output to file
   - `-epss`: show epss score
      - `--epss --epss-score 0.1`: filter for vulnerabilities that have more than 10% probability to be exploited.

#### Troubleshoot
Docker scout does not run because `/tmp` is full:
- `df -h`: to see usage of partitions.
- `du -h /home/schnee/DIA-sysds-scout-tmp/` to see directory usage.
- `docker scout cache df` and `docker scout cache prune` to view and clear scout cache.

Change default tmp partition to a bigger filesystem
- `export TMPDIR=/absolute/path/to/sysds-scout-tmp`
- `export DOCKER_SCOUT_CACHE_DIR=/absolute/path/to/sysds-scout-tmp`

<details>

   <summary> Directly from filesystem (Dockerfile)</summary>

   > Does Not Work

   No vulnerability found

   ```bash
   cd project_root_directory
   docker scout cves --format markdown -o scout_results/output0.md --epss fs://docker_subdirecory/file.Dockerfile
   ```

</details>

### Solve vulnerabilities

By using the CVE code and reading its description, most vulnerabilities have solutions or workarounds.

#### Upgrading the related package

One way to solve a vulerability is simply to upgrade the package the CVE happens in.

> Sometimes, the package that raised the CVE is not included in the `pom.xml`. This can happen if the package is a dependency of another imported package. \
> `mvn dependency:tree` (example output can be found [here](#output-of-mvn-dependencytree-before-any-change)) shows all imported packages and implicit dependencies imported by other packets.

#### Remove or switch the packet

If the vulnerability comes from a transitive dependency:
1. update the parent dependency to a newer version which doesn't use the vulnerable packet. \
This can be verified in Maven Repository <https://mvnrepository.com/> by searching the parent packet.
1. exclude the vulnerable package from the parent and explicitly import a newer version of the vulnerable package. \
By doing this, the parent dependency will use the explicit version instead of the vulnerable one.

### Inspection commands to find tricky packages

As mentioned before, using the `sarif` format in scout could help identify where the vulnerale package comes from.

If this does not help, it is possible to inspect the image directly to find the related jar. Here, `apache/zookeeper` is used as reference:
- copy container filesystem to analyze jars: 

  ```sh
  docker create --name temp apache/systemds
  docker export temp > img.tar
  mkdir filesystem_img
  tar -xf img.tar -C filesystem_img
  #Analysis
  rm -rf img.tar temp filesystem_img
  docker rm temp
  ```
  - Find all references to zookeeper

    `find filesystem_img -name '*zookeeper*'`
  - List all libs 
    
    `ls -la ./filesystem_img/systemds/target/lib/`
  - Inspect the jars in the image for references to zookeeper

    `find ./filesystem_img -name '*.jar' -exec sh -c 'jar tf {} | grep -i zookeeper && echo "--- Found in {}"' \;`
- Inspect the local jars for references to zookeeper 

  `jar tf ~/.m2/repository/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar | grep -i zookeeper`

### `docker scout` helloworld example

To test if docker scout works correctly on your machine, you can create a simple "Hello World" Dockerfile.

```bash
mkdir scout_hello_world
cd scout_hello_world
echo -e "FROM ubuntu:latest\nRUN apt-get update && apt-get install -y curl\nCMD ["curl", "https://example.com"]" > Dockerfile
docker build -t scout_hello_world_img .
# optionally check image: docker images | grep hello_world
docker scout cves scout_hello_world_img
```

1. create and enter new directory
2. Create new file `Dockerfile` and write content into it
1. Build docker image from current directory `.` \
   With name scout_hello_wolrd (`-t`)
1. Scout the image with the name

The output can be viewed in the [appendix](#docker-scout-helloworld-output)


## Student Project SYSTEMDS-3878

### Results

We managed to solve a lot of CVEs: 

| FROM: | <img alt="critical: 4" src="https://img.shields.io/badge/critical-4-8b1924"/>    | <img alt="high: 29" src="https://img.shields.io/badge/high-29-e25d68"/> | <img alt="medium: 36" src="https://img.shields.io/badge/medium-36-fbb552"/> | <img alt="low: 9" src="https://img.shields.io/badge/low-9-fce1a9"/> | <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/> |
| :---- | :------------------------------------------------------------------------------- | :---------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------- |
| TO:   | <img alt="critical: 0" src="https://img.shields.io/badge/critical-0-lightgrey"/> | <img alt="high: 6" src="https://img.shields.io/badge/high-6-e25d68"/>   | <img alt="medium: 8" src="https://img.shields.io/badge/medium-9-fbb552"/>   | <img alt="low: 1" src="https://img.shields.io/badge/low-1-fce1a9"/> | <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/> |

The `scout` reports can be viewed in details: 
- [scan-before-fixes](./scan-before-fixes/README.md)
- [scan-after-fixes](./scan-after-fixes/README.md)

[unfixed-vulnerabilities/README.md](./unfixed-vulnerabilities/README.md) has been written to explain the last vulnerabilities that hove not been fixed and why.

### Summary of Changes

The changes made during the project have been documented in the [summary-of-changes/README.md](./summary-of-changes/README.md). A detailed example on how we solved vulnerabilities is also described.

### Testing
Running `mvn clean verify` after our changes returns the same output.

## Appendix

### `docker scout` helloworld output

[Return](#docker-scout-helloworld-example) to `docker scout` helloworld

<details>
   
   <summary> Output from scout should be something similar to this: </summary>

   ```bash
      ✓ Image stored for indexing
      ✓ Indexed 160 packages
      ✓ Provenance obtained from attestation
      ✗ Detected 8 vulnerable packages with a total of 10 vulnerabilities


   ## Overview

                      │         Analyzed Image          
   ───────────────────┼─────────────────────────────────
    Target            │  scout_hello_world_img:latest   
      digest          │  65884f6905ea                   
      platform        │ linux/amd64                     
      vulnerabilities │    0C     0H     2M     8L      
      size            │ 73 MB                           
      packages        │ 160                             


   ## Packages and Vulnerabilities

      0C     0H     1M     0L  pam 1.5.3-5ubuntu5.5
   pkg:deb/ubuntu/pam@1.5.3-5ubuntu5.5?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ MEDIUM CVE-2025-8941
         https://scout.docker.com/v/CVE-2025-8941
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     1M     0L  tar 1.35+dfsg-3build1
   pkg:deb/ubuntu/tar@1.35%2Bdfsg-3build1?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ MEDIUM CVE-2025-45582
         https://scout.docker.com/v/CVE-2025-45582
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     3L  curl 8.5.0-2ubuntu10.6
   pkg:deb/ubuntu/curl@8.5.0-2ubuntu10.6?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2025-9086
         https://scout.docker.com/v/CVE-2025-9086
         Affected range : >=0        
         Fixed version  : not fixed  
      
      ✗ LOW CVE-2025-10148
         https://scout.docker.com/v/CVE-2025-10148
         Affected range : >=0        
         Fixed version  : not fixed  
      
      ✗ LOW CVE-2025-0167
         https://scout.docker.com/v/CVE-2025-0167
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     1L  libgcrypt20 1.10.3-2build1
   pkg:deb/ubuntu/libgcrypt20@1.10.3-2build1?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2024-2236
         https://scout.docker.com/v/CVE-2024-2236
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     1L  coreutils 9.4-3ubuntu6.1
   pkg:deb/ubuntu/coreutils@9.4-3ubuntu6.1?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2016-2781
         https://scout.docker.com/v/CVE-2016-2781
         Affected range : >=0                                           
         Fixed version  : not fixed                                     
         CVSS Score     : 6.5                                           
         CVSS Vector    : CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:C/C:N/I:H/A:N  
      

      0C     0H     0M     1L  shadow 1:4.13+dfsg1-4ubuntu3.2
   pkg:deb/ubuntu/shadow@1%3A4.13%2Bdfsg1-4ubuntu3.2?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2024-56433
         https://scout.docker.com/v/CVE-2024-56433
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     1L  gnupg2 2.4.4-2ubuntu17.3
   pkg:deb/ubuntu/gnupg2@2.4.4-2ubuntu17.3?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2022-3219
         https://scout.docker.com/v/CVE-2022-3219
         Affected range : >=0                                           
         Fixed version  : not fixed                                     
         CVSS Score     : 3.3                                           
         CVSS Vector    : CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:L  
      

      0C     0H     0M     1L  openssl 3.0.13-0ubuntu3.6
   pkg:deb/ubuntu/openssl@3.0.13-0ubuntu3.6?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2024-41996
         https://scout.docker.com/v/CVE-2024-41996
         Affected range : >=0        
         Fixed version  : not fixed  
      


   10 vulnerabilities found in 8 packages
   CRITICAL  0  
   HIGH      0  
   MEDIUM    2  
   LOW       8  


   What's next:
      View base image update recommendations → docker scout recommendations scout_hello_world_img:latest


   ```

</details>

### Output of `mvn dependency:tree` before any change

[Return](#upgrading-the-related-package) to solving vulnerabilities

<details>

   <summary>terminal output</summary>

   ```bash
   [INFO] org.apache.systemds:systemds:jar:3.4.0-SNAPSHOT
   [INFO] +- org.jcuda:jcuda:jar:12.6.0:provided
   [INFO] +- org.jcuda:jcublas:jar:12.6.0:provided
   [INFO] +- org.jcuda:jcusparse:jar:12.6.0:provided
   [INFO] +- org.jcuda:jcusolver:jar:12.6.0:provided
   [INFO] +- org.jcuda:jcudnn:jar:12.6.0:provided
   [INFO] +- org.jcuda:jcuda-natives:jar:windows-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcublas-natives:jar:windows-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcusparse-natives:jar:windows-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcusolver-natives:jar:windows-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcudnn-natives:jar:windows-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcuda-natives:jar:linux-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcublas-natives:jar:linux-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcusparse-natives:jar:linux-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcusolver-natives:jar:linux-x86_64:12.6.0:provided
   [INFO] +- org.jcuda:jcudnn-natives:jar:linux-x86_64:12.6.0:provided
   [INFO] +- org.apache.spark:spark-core_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.avro:avro:jar:1.11.2:compile
   [INFO] |  +- org.apache.avro:avro-mapred:jar:1.11.2:compile
   [INFO] |  |  \- org.apache.avro:avro-ipc:jar:1.11.2:compile
   [INFO] |  |     \- org.tukaani:xz:jar:1.9:compile
   [INFO] |  +- com.twitter:chill_2.12:jar:0.10.0:compile
   [INFO] |  |  \- com.esotericsoftware:kryo-shaded:jar:4.0.2:compile
   [INFO] |  |     \- com.esotericsoftware:minlog:jar:1.3.0:compile
   [INFO] |  +- com.twitter:chill-java:jar:0.10.0:compile
   [INFO] |  +- org.apache.xbean:xbean-asm9-shaded:jar:4.23:compile
   [INFO] |  +- org.apache.spark:spark-launcher_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.spark:spark-kvstore_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.spark:spark-network-common_2.12:jar:3.5.0:compile
   [INFO] |  |  \- com.google.crypto.tink:tink:jar:1.9.0:compile
   [INFO] |  |     \- joda-time:joda-time:jar:2.12.5:compile
   [INFO] |  +- org.apache.spark:spark-network-shuffle_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.spark:spark-unsafe_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.spark:spark-common-utils_2.12:jar:3.5.0:compile
   [INFO] |  +- javax.activation:activation:jar:1.1.1:compile
   [INFO] |  +- org.apache.curator:curator-recipes:jar:2.13.0:compile
   [INFO] |  |  \- org.apache.curator:curator-framework:jar:2.13.0:compile
   [INFO] |  +- org.apache.zookeeper:zookeeper:jar:3.6.3:compile
   [INFO] |  |  +- org.apache.zookeeper:zookeeper-jute:jar:3.6.3:compile
   [INFO] |  |  \- org.apache.yetus:audience-annotations:jar:0.5.0:compile
   [INFO] |  +- jakarta.servlet:jakarta.servlet-api:jar:4.0.3:compile
   [INFO] |  +- commons-codec:commons-codec:jar:1.16.0:compile
   [INFO] |  +- org.apache.commons:commons-compress:jar:1.23.0:compile
   [INFO] |  +- org.apache.commons:commons-lang3:jar:3.12.0:compile
   [INFO] |  +- org.apache.commons:commons-text:jar:1.10.0:compile
   [INFO] |  +- commons-io:commons-io:jar:2.13.0:compile
   [INFO] |  +- commons-collections:commons-collections:jar:3.2.2:compile
   [INFO] |  +- org.apache.commons:commons-collections4:jar:4.4:compile
   [INFO] |  +- com.google.code.findbugs:jsr305:jar:3.0.0:compile
   [INFO] |  +- com.ning:compress-lzf:jar:1.1.2:compile
   [INFO] |  +- org.xerial.snappy:snappy-java:jar:1.1.10.3:compile
   [INFO] |  +- org.lz4:lz4-java:jar:1.8.0:compile
   [INFO] |  +- com.github.luben:zstd-jni:jar:1.5.5-4:compile
   [INFO] |  +- org.roaringbitmap:RoaringBitmap:jar:0.9.45:compile
   [INFO] |  |  \- org.roaringbitmap:shims:jar:0.9.45:runtime
   [INFO] |  +- org.scala-lang.modules:scala-xml_2.12:jar:2.1.0:compile
   [INFO] |  +- org.scala-lang:scala-library:jar:2.12.18:compile
   [INFO] |  +- org.scala-lang:scala-reflect:jar:2.12.18:compile
   [INFO] |  +- org.json4s:json4s-jackson_2.12:jar:3.7.0-M11:compile
   [INFO] |  |  \- org.json4s:json4s-core_2.12:jar:3.7.0-M11:compile
   [INFO] |  |     +- org.json4s:json4s-ast_2.12:jar:3.7.0-M11:compile
   [INFO] |  |     \- org.json4s:json4s-scalap_2.12:jar:3.7.0-M11:compile
   [INFO] |  +- org.glassfish.jersey.core:jersey-client:jar:2.40:compile
   [INFO] |  |  +- jakarta.ws.rs:jakarta.ws.rs-api:jar:2.1.6:compile
   [INFO] |  |  \- org.glassfish.hk2.external:jakarta.inject:jar:2.6.1:compile
   [INFO] |  +- org.glassfish.jersey.core:jersey-common:jar:2.40:compile
   [INFO] |  |  +- jakarta.annotation:jakarta.annotation-api:jar:1.3.5:compile
   [INFO] |  |  \- org.glassfish.hk2:osgi-resource-locator:jar:1.0.3:compile
   [INFO] |  +- org.glassfish.jersey.core:jersey-server:jar:2.40:compile
   [INFO] |  |  \- jakarta.validation:jakarta.validation-api:jar:2.0.2:compile
   [INFO] |  +- org.glassfish.jersey.containers:jersey-container-servlet:jar:2.40:compile
   [INFO] |  +- org.glassfish.jersey.containers:jersey-container-servlet-core:jar:2.40:compile
   [INFO] |  +- org.glassfish.jersey.inject:jersey-hk2:jar:2.40:compile
   [INFO] |  |  +- org.glassfish.hk2:hk2-locator:jar:2.6.1:compile
   [INFO] |  |  |  +- org.glassfish.hk2.external:aopalliance-repackaged:jar:2.6.1:compile
   [INFO] |  |  |  +- org.glassfish.hk2:hk2-api:jar:2.6.1:compile
   [INFO] |  |  |  \- org.glassfish.hk2:hk2-utils:jar:2.6.1:compile
   [INFO] |  |  \- org.javassist:javassist:jar:3.29.2-GA:compile
   [INFO] |  +- io.netty:netty-transport-native-epoll:jar:linux-x86_64:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-native-epoll:jar:linux-aarch_64:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-native-kqueue:jar:osx-aarch_64:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-native-kqueue:jar:osx-x86_64:4.1.96.Final:compile
   [INFO] |  +- com.clearspring.analytics:stream:jar:2.9.6:compile
   [INFO] |  +- io.dropwizard.metrics:metrics-core:jar:4.2.19:compile
   [INFO] |  +- io.dropwizard.metrics:metrics-jvm:jar:4.2.19:compile
   [INFO] |  +- io.dropwizard.metrics:metrics-json:jar:4.2.19:compile
   [INFO] |  +- io.dropwizard.metrics:metrics-graphite:jar:4.2.19:compile
   [INFO] |  +- io.dropwizard.metrics:metrics-jmx:jar:4.2.19:compile
   [INFO] |  +- com.fasterxml.jackson.module:jackson-module-scala_2.12:jar:2.15.2:compile
   [INFO] |  |  \- com.thoughtworks.paranamer:paranamer:jar:2.8:compile
   [INFO] |  +- org.apache.ivy:ivy:jar:2.5.1:compile
   [INFO] |  +- oro:oro:jar:2.0.8:compile
   [INFO] |  +- net.razorvine:pickle:jar:1.3:compile
   [INFO] |  +- org.apache.spark:spark-tags_2.12:jar:3.5.0:compile
   [INFO] |  \- org.apache.commons:commons-crypto:jar:1.1.0:compile
   [INFO] +- org.apache.spark:spark-sql_2.12:jar:3.5.0:compile
   [INFO] |  +- org.rocksdb:rocksdbjni:jar:8.3.2:compile
   [INFO] |  +- com.univocity:univocity-parsers:jar:2.9.1:compile
   [INFO] |  +- org.apache.spark:spark-sketch_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.spark:spark-catalyst_2.12:jar:3.5.0:compile
   [INFO] |  |  +- org.apache.spark:spark-sql-api_2.12:jar:3.5.0:compile
   [INFO] |  |  |  +- org.apache.arrow:arrow-vector:jar:12.0.1:compile
   [INFO] |  |  |  |  +- org.apache.arrow:arrow-format:jar:12.0.1:compile
   [INFO] |  |  |  |  +- org.apache.arrow:arrow-memory-core:jar:12.0.1:compile
   [INFO] |  |  |  |  +- com.fasterxml.jackson.datatype:jackson-datatype-jsr310:jar:2.15.1:compile
   [INFO] |  |  |  |  \- com.google.flatbuffers:flatbuffers-java:jar:1.12.0:compile
   [INFO] |  |  |  \- org.apache.arrow:arrow-memory-netty:jar:12.0.1:compile
   [INFO] |  |  \- org.apache.datasketches:datasketches-java:jar:3.3.0:compile
   [INFO] |  |     \- org.apache.datasketches:datasketches-memory:jar:2.1.0:compile
   [INFO] |  +- org.apache.orc:orc-core:jar:shaded-protobuf:1.9.1:compile
   [INFO] |  |  +- org.apache.orc:orc-shims:jar:1.9.1:compile
   [INFO] |  |  +- io.airlift:aircompressor:jar:0.25:compile
   [INFO] |  |  +- org.jetbrains:annotations:jar:17.0.0:compile
   [INFO] |  |  \- org.threeten:threeten-extra:jar:1.7.1:compile
   [INFO] |  +- org.apache.orc:orc-mapreduce:jar:shaded-protobuf:1.9.1:compile
   [INFO] |  +- org.apache.hive:hive-storage-api:jar:2.8.1:compile
   [INFO] |  +- org.apache.parquet:parquet-column:jar:1.13.1:compile
   [INFO] |  |  +- org.apache.parquet:parquet-common:jar:1.13.1:compile
   [INFO] |  |  \- org.apache.parquet:parquet-encoding:jar:1.13.1:compile
   [INFO] |  \- org.apache.parquet:parquet-hadoop:jar:1.13.1:compile
   [INFO] |     +- org.apache.parquet:parquet-format-structures:jar:1.13.1:compile
   [INFO] |     \- org.apache.parquet:parquet-jackson:jar:1.13.1:runtime
   [INFO] +- org.apache.spark:spark-mllib_2.12:jar:3.5.0:compile
   [INFO] |  +- org.scala-lang.modules:scala-parser-combinators_2.12:jar:2.3.0:compile
   [INFO] |  +- org.apache.spark:spark-streaming_2.12:jar:3.5.0:compile
   [INFO] |  +- org.apache.spark:spark-graphx_2.12:jar:3.5.0:compile
   [INFO] |  |  \- net.sourceforge.f2j:arpack_combined_all:jar:0.1:compile
   [INFO] |  +- org.apache.spark:spark-mllib-local_2.12:jar:3.5.0:compile
   [INFO] |  +- org.scalanlp:breeze_2.12:jar:2.1.0:compile
   [INFO] |  |  +- org.scalanlp:breeze-macros_2.12:jar:2.1.0:compile
   [INFO] |  |  +- net.sf.opencsv:opencsv:jar:2.3:compile
   [INFO] |  |  +- com.github.wendykierp:JTransforms:jar:3.1:compile
   [INFO] |  |  |  \- pl.edu.icm:JLargeArrays:jar:1.5:compile
   [INFO] |  |  +- org.scala-lang.modules:scala-collection-compat_2.12:jar:2.7.0:compile
   [INFO] |  |  \- org.typelevel:spire_2.12:jar:0.17.0:compile
   [INFO] |  |     +- org.typelevel:spire-macros_2.12:jar:0.17.0:compile
   [INFO] |  |     +- org.typelevel:spire-platform_2.12:jar:0.17.0:compile
   [INFO] |  |     +- org.typelevel:spire-util_2.12:jar:0.17.0:compile
   [INFO] |  |     \- org.typelevel:algebra_2.12:jar:2.0.1:compile
   [INFO] |  |        \- org.typelevel:cats-kernel_2.12:jar:2.1.1:compile
   [INFO] |  +- org.glassfish.jaxb:jaxb-runtime:jar:2.3.2:compile
   [INFO] |  |  +- jakarta.xml.bind:jakarta.xml.bind-api:jar:2.3.2:compile
   [INFO] |  |  \- com.sun.istack:istack-commons-runtime:jar:3.0.8:compile
   [INFO] |  +- dev.ludovic.netlib:blas:jar:3.0.3:compile
   [INFO] |  +- dev.ludovic.netlib:lapack:jar:3.0.3:compile
   [INFO] |  \- dev.ludovic.netlib:arpack:jar:3.0.3:compile
   [INFO] +- org.apache.hadoop:hadoop-common:jar:3.3.6:compile
   [INFO] |  +- org.apache.hadoop.thirdparty:hadoop-shaded-protobuf_3_7:jar:1.1.1:compile
   [INFO] |  +- org.apache.hadoop:hadoop-annotations:jar:3.3.6:compile
   [INFO] |  +- org.apache.hadoop.thirdparty:hadoop-shaded-guava:jar:1.1.1:compile
   [INFO] |  +- com.google.guava:guava:jar:27.0-jre:compile
   [INFO] |  |  +- com.google.guava:failureaccess:jar:1.0:compile
   [INFO] |  |  +- com.google.guava:listenablefuture:jar:9999.0-empty-to-avoid-conflict-with-guava:compile
   [INFO] |  |  +- org.checkerframework:checker-qual:jar:2.5.2:compile
   [INFO] |  |  \- org.codehaus.mojo:animal-sniffer-annotations:jar:1.17:compile
   [INFO] |  +- commons-cli:commons-cli:jar:1.2:compile
   [INFO] |  +- org.apache.httpcomponents:httpclient:jar:4.5.13:compile
   [INFO] |  |  \- org.apache.httpcomponents:httpcore:jar:4.4.13:compile
   [INFO] |  +- commons-net:commons-net:jar:3.9.0:compile
   [INFO] |  +- javax.servlet:javax.servlet-api:jar:3.1.0:compile
   [INFO] |  +- jakarta.activation:jakarta.activation-api:jar:1.2.1:compile
   [INFO] |  +- org.eclipse.jetty:jetty-server:jar:9.4.51.v20230217:compile
   [INFO] |  |  +- org.eclipse.jetty:jetty-http:jar:9.4.51.v20230217:compile
   [INFO] |  |  \- org.eclipse.jetty:jetty-io:jar:9.4.51.v20230217:compile
   [INFO] |  +- org.eclipse.jetty:jetty-util:jar:9.4.51.v20230217:compile
   [INFO] |  +- org.eclipse.jetty:jetty-servlet:jar:9.4.51.v20230217:compile
   [INFO] |  |  \- org.eclipse.jetty:jetty-security:jar:9.4.51.v20230217:compile
   [INFO] |  +- org.eclipse.jetty:jetty-webapp:jar:9.4.51.v20230217:compile
   [INFO] |  |  \- org.eclipse.jetty:jetty-xml:jar:9.4.51.v20230217:compile
   [INFO] |  +- javax.servlet.jsp:jsp-api:jar:2.1:runtime
   [INFO] |  +- com.sun.jersey:jersey-core:jar:1.19.4:compile
   [INFO] |  |  \- javax.ws.rs:jsr311-api:jar:1.1.1:compile
   [INFO] |  +- com.sun.jersey:jersey-servlet:jar:1.19.4:compile
   [INFO] |  +- com.github.pjfanning:jersey-json:jar:1.20:compile
   [INFO] |  |  +- org.codehaus.jettison:jettison:jar:1.1:compile
   [INFO] |  |  \- com.sun.xml.bind:jaxb-impl:jar:2.2.3-1:compile
   [INFO] |  +- com.sun.jersey:jersey-server:jar:1.19.4:compile
   [INFO] |  +- ch.qos.reload4j:reload4j:jar:1.2.22:compile
   [INFO] |  +- commons-beanutils:commons-beanutils:jar:1.9.4:compile
   [INFO] |  +- org.apache.commons:commons-configuration2:jar:2.8.0:compile
   [INFO] |  +- com.google.re2j:re2j:jar:1.1:compile
   [INFO] |  +- com.google.code.gson:gson:jar:2.9.0:compile
   [INFO] |  +- org.apache.hadoop:hadoop-auth:jar:3.3.6:compile
   [INFO] |  |  +- com.nimbusds:nimbus-jose-jwt:jar:9.8.1:compile
   [INFO] |  |  \- org.apache.kerby:kerb-simplekdc:jar:1.0.1:compile
   [INFO] |  |     +- org.apache.kerby:kerb-client:jar:1.0.1:compile
   [INFO] |  |     |  +- org.apache.kerby:kerby-config:jar:1.0.1:compile
   [INFO] |  |     |  +- org.apache.kerby:kerb-common:jar:1.0.1:compile
   [INFO] |  |     |  |  \- org.apache.kerby:kerb-crypto:jar:1.0.1:compile
   [INFO] |  |     |  +- org.apache.kerby:kerb-util:jar:1.0.1:compile
   [INFO] |  |     |  \- org.apache.kerby:token-provider:jar:1.0.1:compile
   [INFO] |  |     \- org.apache.kerby:kerb-admin:jar:1.0.1:compile
   [INFO] |  |        +- org.apache.kerby:kerb-server:jar:1.0.1:compile
   [INFO] |  |        |  \- org.apache.kerby:kerb-identity:jar:1.0.1:compile
   [INFO] |  |        \- org.apache.kerby:kerby-xdr:jar:1.0.1:compile
   [INFO] |  +- com.jcraft:jsch:jar:0.1.55:compile
   [INFO] |  +- org.apache.curator:curator-client:jar:5.2.0:compile
   [INFO] |  +- org.apache.kerby:kerb-core:jar:1.0.1:compile
   [INFO] |  |  \- org.apache.kerby:kerby-pkix:jar:1.0.1:compile
   [INFO] |  |     +- org.apache.kerby:kerby-asn1:jar:1.0.1:compile
   [INFO] |  |     \- org.apache.kerby:kerby-util:jar:1.0.1:compile
   [INFO] |  +- org.codehaus.woodstox:stax2-api:jar:4.2.1:compile
   [INFO] |  +- com.fasterxml.woodstox:woodstox-core:jar:5.4.0:compile
   [INFO] |  \- dnsjava:dnsjava:jar:2.1.7:compile
   [INFO] +- org.apache.hadoop:hadoop-hdfs:jar:3.3.6:compile
   [INFO] |  +- org.eclipse.jetty:jetty-util-ajax:jar:9.4.51.v20230217:compile
   [INFO] |  +- commons-daemon:commons-daemon:jar:1.0.13:compile
   [INFO] |  +- io.netty:netty:jar:3.10.6.Final:compile
   [INFO] |  \- org.fusesource.leveldbjni:leveldbjni-all:jar:1.8:compile
   [INFO] +- org.apache.hadoop:hadoop-client:jar:3.3.6:compile
   [INFO] |  +- org.apache.hadoop:hadoop-hdfs-client:jar:3.3.6:compile
   [INFO] |  +- org.apache.hadoop:hadoop-yarn-api:jar:3.3.6:compile
   [INFO] |  |  \- javax.xml.bind:jaxb-api:jar:2.2.11:compile
   [INFO] |  +- org.apache.hadoop:hadoop-yarn-client:jar:3.3.6:compile
   [INFO] |  |  +- org.eclipse.jetty.websocket:websocket-client:jar:9.4.51.v20230217:compile
   [INFO] |  |  |  +- org.eclipse.jetty:jetty-client:jar:9.4.51.v20230217:compile
   [INFO] |  |  |  \- org.eclipse.jetty.websocket:websocket-common:jar:9.4.51.v20230217:compile
   [INFO] |  |  |     \- org.eclipse.jetty.websocket:websocket-api:jar:9.4.51.v20230217:compile
   [INFO] |  |  \- org.jline:jline:jar:3.9.0:compile
   [INFO] |  +- org.apache.hadoop:hadoop-mapreduce-client-core:jar:3.3.6:compile
   [INFO] |  |  \- org.apache.hadoop:hadoop-yarn-common:jar:3.3.6:compile
   [INFO] |  |     +- com.sun.jersey:jersey-client:jar:1.19.4:compile
   [INFO] |  |     +- com.fasterxml.jackson.module:jackson-module-jaxb-annotations:jar:2.12.7:compile
   [INFO] |  |     \- com.fasterxml.jackson.jaxrs:jackson-jaxrs-json-provider:jar:2.12.7:compile
   [INFO] |  |        \- com.fasterxml.jackson.jaxrs:jackson-jaxrs-base:jar:2.12.7:compile
   [INFO] |  \- org.apache.hadoop:hadoop-mapreduce-client-jobclient:jar:3.3.6:compile
   [INFO] |     \- org.apache.hadoop:hadoop-mapreduce-client-common:jar:3.3.6:compile
   [INFO] +- commons-logging:commons-logging:jar:1.1.3:compile
   [INFO] +- org.apache.commons:commons-math3:jar:3.4.1:compile
   [INFO] +- org.apache.wink:wink-json4j:jar:1.4:compile
   [INFO] +- com.fasterxml.jackson.core:jackson-databind:jar:2.15.2:compile
   [INFO] |  +- com.fasterxml.jackson.core:jackson-annotations:jar:2.15.2:compile
   [INFO] |  \- com.fasterxml.jackson.core:jackson-core:jar:2.15.2:compile
   [INFO] +- junit:junit:jar:4.13.1:provided
   [INFO] |  \- org.hamcrest:hamcrest-core:jar:1.3:provided
   [INFO] +- org.openjdk.jol:jol-core:jar:0.10:test
   [INFO] +- org.mockito:mockito-core:jar:5.1.0:test
   [INFO] |  +- net.bytebuddy:byte-buddy:jar:1.12.22:test
   [INFO] |  +- net.bytebuddy:byte-buddy-agent:jar:1.12.22:test
   [INFO] |  \- org.objenesis:objenesis:jar:3.3:compile
   [INFO] +- com.github.stephenc.jcip:jcip-annotations:jar:1.0-1:test
   [INFO] +- org.codehaus.janino:janino:jar:3.1.9:provided
   [INFO] |  \- org.codehaus.janino:commons-compiler:jar:3.1.9:compile
   [INFO] +- org.antlr:antlr4:jar:4.8:provided
   [INFO] |  +- org.antlr:ST4:jar:4.3:provided
   [INFO] |  +- org.abego.treelayout:org.abego.treelayout.core:jar:1.0.3:provided
   [INFO] |  +- org.glassfish:javax.json:jar:1.0.4:provided
   [INFO] |  \- com.ibm.icu:icu4j:jar:61.1:provided
   [INFO] +- org.antlr:antlr4-runtime:jar:4.8:compile
   [INFO] +- org.apache.derby:derby:jar:10.14.2.0:provided
   [INFO] +- io.netty:netty-all:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-buffer:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-dns:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-haproxy:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-http:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-http2:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-memcache:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-mqtt:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-redis:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-smtp:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-socks:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-stomp:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-codec-xml:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-common:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-handler:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-native-unix-common:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-handler-proxy:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-handler-ssl-ocsp:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-resolver:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-resolver-dns:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-rxtx:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-sctp:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-udt:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-classes-epoll:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-transport-classes-kqueue:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-resolver-dns-classes-macos:jar:4.1.96.Final:compile
   [INFO] |  +- io.netty:netty-resolver-dns-native-macos:jar:osx-x86_64:4.1.96.Final:runtime
   [INFO] |  \- io.netty:netty-resolver-dns-native-macos:jar:osx-aarch_64:4.1.96.Final:runtime
   [INFO] +- net.sf.py4j:py4j:jar:0.10.9:compile
   [INFO] +- com.google.protobuf:protobuf-java:jar:3.23.4:compile
   [INFO] +- com.google.protobuf:protobuf-java-util:jar:3.23.4:compile
   [INFO] |  +- com.google.errorprone:error_prone_annotations:jar:2.18.0:compile
   [INFO] |  \- com.google.j2objc:j2objc-annotations:jar:2.8:compile
   [INFO] +- org.slf4j:slf4j-api:jar:2.0.11:compile
   [INFO] +- org.slf4j:slf4j-reload4j:jar:2.0.11:compile
   [INFO] +- org.slf4j:jul-to-slf4j:jar:2.0.11:compile
   [INFO] +- org.slf4j:jcl-over-slf4j:jar:2.0.11:compile
   [INFO] +- org.apache.logging.log4j:log4j-api:jar:2.22.1:compile
   [INFO] +- org.apache.logging.log4j:log4j-core:jar:2.22.1:compile
   [INFO] \- ch.randelshofer:fastdoubleparser:jar:0.9.0:compile
   [INFO] ------------------------------------------------------------------------
   [INFO] BUILD SUCCESS
   [INFO] ------------------------------------------------------------------------
   [INFO] Total time:  59.447 s
   [INFO] Finished at: 2025-12-08T10:24:00+01:00
   [INFO] ------------------------------------------------------------------------
   ```
</details>
