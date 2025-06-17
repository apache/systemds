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

FROM alpine:latest

WORKDIR /usr/src/

# Do basic updates on the image
RUN apk add --update --no-cache \
		wget \
		git \
		ca-certificates \
		openjdk17 \
		bash

# Set environment variables
# Maven
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn
# Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH
ENV SYSTEMDS_ROOT=/usr/src/systemds
ENV PATH=$SYSTEMDS_ROOT/bin:$PATH
ENV SYSDS_QUIET=1

# Download Java and Mvn 
RUN wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn

# Build the system
RUN git clone --depth 1 https://github.com/apache/systemds.git systemds && \
	cd /usr/src/systemds/ && \
	mvn --no-transfer-progress clean package -P distribution

# Remove all unnecessary files from the Image
RUN	rm -rf .git && \
	rm -rf .github && \
	rm -rf target/javadoc** && \
	rm -rf target/apidocs** && \
	rm -rf target/classes && \
	rm -rf target/test-classes && \
	rm -rf target/maven-archiver && \
	rm -rf target/systemds-** && \
	rm -rf docs && \
	rm -rf src && \
	rm -rf /usr/lib/mvn && \
	rm -rf CONTRIBUTING.md && \
	rm -rf pom.xml && \ 
	rm -rf ~/.m2

COPY docker/mountFolder/main.dml /input/main.dml

CMD ["systemds", "/input/main.dml"]
