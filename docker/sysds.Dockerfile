#-------------------------------------------------------------
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
#-------------------------------------------------------------

FROM ubuntu:20.04

WORKDIR /usr/src/

# Maven
ENV MAVEN_VERSION 3.8.3
ENV MAVEN_HOME /usr/lib/mvn
ENV PATH $MAVEN_HOME/bin:$PATH
# Java
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH
ENV SYSTEMDS_ROOT=/usr/src/systemds
ENV PATH $SYSTEMDS_ROOT/bin:$PATH
ENV SYSDS_QUIET=1

RUN apt-get update -qq \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
	wget \
	git \
	ca-certificates \
	&& apt-get clean \
	&& mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/AdoptOpenJDK/openjdk11-upstream-binaries/releases/download/jdk-11.0.13%2B8/OpenJDK11U-jdk_x64_linux_11.0.13_8.tar.gz | tar xzf - \
	&& mv openjdk-11.0.13_8 /usr/lib/jvm/java-11-openjdk-amd64 \
	&& wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn \
	&& git clone --depth 1 https://github.com/apache/systemds.git systemds && \
	cd /usr/src/systemds/ && \
	mvn --no-transfer-progress clean package -P distribution && \
	rm -r .git && \
	rm -r .github && \
	rm -r target/javadoc** && \
	rm -r target/apidocs** && \
	rm -r target/classes && \
	rm -r target/test-classes && \
	rm -r target/hadoop-test && \
	rm -r target/maven-archiver && \
	rm -r target/systemds-** && \
	rm -r docs && \
	rm -r src && \
	rm -r /usr/lib/mvn && \
	rm -r CONTRIBUTING.md && \
	rm -r pom.xml && \ 
	rm -r ~/.m2


COPY docker/mountFolder/main.dml /input/main.dml

CMD ["systemds", "/input/main.dml"]
