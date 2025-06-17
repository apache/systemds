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

FROM alpine:3.20@sha256:de4fe7064d8f98419ea6b49190df1abbf43450c1702eeb864fe9ced453c1cc5f AS compile-image

WORKDIR /usr/src/

# Do basic updates on the image
RUN apk add --no-cache \
		wget \
		git \
		ca-certificates \
		bash

# Set environment variables
# Maven
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn
# Java
ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH
ENV SYSTEMDS_ROOT=/usr/src/systemds
ENV PATH=$SYSTEMDS_ROOT/bin:$PATH
ENV SYSDS_QUIET=1

# Download Mvn and JDK
RUN mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jdk_x64_alpine-linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6 $JAVA_HOME \
	&& wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn

# Build the system
RUN git clone -b SYSTEMDS-3877 --depth 1 https://github.com/alexanderschmi/systemds.git systemds && \
	cd /usr/src/systemds/ && \
	mvn --no-transfer-progress clean package -P distribution

COPY docker/mountFolder/main.dml /input/main.dml

# Remove all unnecessary files from the Image
RUN	cd /usr/src/systemds/ && \
	rm -rf .git && \
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
	rm -rf ~/.m2 && \
	rm -rf docker && \
	rm -rf .mvn && \
	rm -rf LICENSE

FROM alpine:3.20@sha256:de4fe7064d8f98419ea6b49190df1abbf43450c1702eeb864fe9ced453c1cc5f

RUN apk add --no-cache bash

ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$PATH
ENV SYSTEMDS_ROOT=/systemds
ENV PATH=$SYSTEMDS_ROOT/bin:$PATH
ENV SYSDS_QUIET=1

RUN mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jre_x64_alpine-linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6-jre $JAVA_HOME

COPY --from=compile-image /usr/src/systemds /systemds
COPY --from=compile-image /input/main.dml /input/main.dml

WORKDIR /input

ENTRYPOINT ["systemds"]
CMD ["main.dml"]
