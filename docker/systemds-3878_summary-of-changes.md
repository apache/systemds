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
    `mvn dependency:tree` shows all implicit dependencies (see the [appendix](#output-of-mvn-dependencytree) for the output). \
    The responsible dependency seems to be the apache spark-core package.

    ```bash
    [INFO] org.apache.systemds:systemds:jar:3.4.0-SNAPSHOT
    [INFO] +- org.apache.spark:spark-core_2.12:jar:3.5.0:compile
    [INFO] |  +- org.apache.zookeeper:zookeeper:jar:3.6.3:compile
    ```

    From the documentation [Spark Project Core](https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.13/4.0.1): Version `3.5.x` uses zookeeper `3.6.3`. \
    The version `4.0.x` uses zookeeper `3.9.3` which still has a known vulnerability.

    Spark-core `4.1.0` uses the latest version `3.9.4` of zookeeper.

    By inspecting the jars, it has been found out that zookeeper is also used by `hadoop-common`
  - Resolve Vulnerability

    Two ideas to solve the vulnerability: 

    - Upgrade `spark-core`
      - `3.4.x` doesn't use scala `2.12` anymore \
        Update other dependencies (`spark-sql` etc) for the whole project to use scala `2.13`. \
        scala `2.13` is not backwards compatible with `2.12`.
      - No direct backwards compatibility between `4.0` and `3.5` \
        [core migration guide](https://spark.apache.org/docs/latest/core-migration-guide.html#upgrading-from-core-35-to-40)
    - Make `spark-core` use zookeeper `3.8.3+`
      - as `spark-core` is a package  using `zookeeper`, explicitely declaring another version of the zookeeper dependency will override spark's dependecy. As zookeeper is backwards compatible, this should not cause problems for the app
      - `hadoop-common 3.3.6` also uses `zookeeper 3.6.3`.

#### Solution

Zookeeper `3.9.4` will be used to solve all zookeeper-related vulnerabilities. `netty-handler` is also explicitly imported to solve a vulnerability related to the implicit version.

**Changes** 
- `pom.xml`
  - explicit dependency \
    `zookeeper 3.9.4` \
    The vulnerability is solved by upgrading the version of `zookeeper`
    ```xml
    <!-- 
    Explicit declaration to use 3.9.4 instead of implicit use of 3.6.3 by spark-core
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
- `sysds.Dockerfile` 

  systemds is cloned from github instead of using the local systemds project.
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

**Verification**
- `docker scout` \
  The critical vulnerability in `zookeeper` is not shown anymore
- `mvn dependency:tree` \
  shows a more recent version of zookeeper

  No packet imported by zookeeper generates CVES.
- building the project works \
  `mvn clean package -P distribution`

### `io.netty/netty@3.10.6.Final`

This is a transitive dependency from `hadoop-hdfs 3.3.6`. Version `3.4.2` does not use the `netty` dependency anymore.

The only change necessary in the code is in the import of the netty package. What was
```java
import org.jboss.netty.handler.codec.compression.CompressionException;
```
became
```java
import io.netty.handler.codec.compression.CompressionException;
```


### toolbox

- Docker scout: 

  `docker scout cves --details --format markdown -o docker/scout_results/sysds_outputX.md --epss apache/systemds`

  docker scout can also show the path to the vulnerability with the `--format sarif` option.
  - Docker scout does not run because `/tmp` is full: \
    `df -h`: to see usage of partitions. \
    `du -h /home/schnee/DIA-sysds-scout-tmp/` to see directory usage. \
    `docker scout cache df` and `docker scout cache prune` to view and clear scout cache.
    
    Change default tmp partition to a bigger filesystem \
    `export TMPDIR=/home/schnee/DIA-sysds-scout-tmp` \
    `export DOCKER_SCOUT_CACHE_DIR=/home/schnee/DIA-sysds-scout-tmp`
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
- Inspect the given jar for references to zookeeper 

  `jar tf ~/.m2/repository/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar | grep -i zookeeper`


## Results

We managed to solve a lot of CVEs: \
FROM: <img alt="critical: 4" src="https://img.shields.io/badge/critical-4-8b1924"/> <img alt="high: 29" src="https://img.shields.io/badge/high-29-e25d68"/> <img alt="medium: 36" src="https://img.shields.io/badge/medium-36-fbb552"/> <img alt="low: 9" src="https://img.shields.io/badge/low-9-fce1a9"/> <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/> \
TO: <img alt="critical: 0" src="https://img.shields.io/badge/critical-0-lightgrey"/> <img alt="high: 6" src="https://img.shields.io/badge/high-6-e25d68"/> <img alt="medium: 8" src="https://img.shields.io/badge/medium-9-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/low-1-fce1a9"/> <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/>

Running `mvn clean verify` after our changes returns the same output.

## Changes

### `pom.xml`

### `sysds.Dockerfile`

#### Local build
In order to run the alanysis tool locally, we changed the Dockerfile to build not from the git apache/systemds:latest, but to build from a copy of the local filesystem:

```Dockerfile
# Build the system
# RUN git clone --depth 1 https://github.com/apache/systemds.git systemds && \
# 	cd /usr/src/systemds/ && \
# 	mvn --no-transfer-progress clean package -P distribution
#
# Copy the local SystemDS source into the image
COPY . /usr/src/systemds
# Build SystemDS
RUN cd /usr/src/systemds && \
    mvn --no-transfer-progress clean package -P distribution
```

This change was reset before merging.

#### `apk` vulnerabilities



## Appendix

### Output of `mvn dependency:tree` before any change

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
