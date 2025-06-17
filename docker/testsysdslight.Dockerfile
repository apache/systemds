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

FROM alpine:3.20@sha256:de4fe7064d8f98419ea6b49190df1abbf43450c1702eeb864fe9ced453c1cc5f

WORKDIR /usr/src/
# Set environment variables
# Maven
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn
# Java
ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/

RUN apk --no-cache add \
	curl-dev \
	libxml2-dev \
	libc6-compat \
	gnupg \
	wget \
	ca-certificates \
	git \
	cmake \
	patchelf \
	R \
	R-dev \
	R-doc \
	build-base \
	&& mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jdk_x64_alpine-linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6 $JAVA_HOME \
	&& wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn


# Install R packages
COPY ./src/test/scripts/installDependencies.R installDependencies.R
RUN Rscript installDependencies.R \
    && rm -f installDependencies.R

# Install SEAL
RUN wget -qO- https://github.com/microsoft/SEAL/archive/refs/tags/v3.7.0.tar.gz | tar xzf - \
    && cd SEAL-3.7.0 \
    && cmake -S . -B build -DBUILD_SHARED_LIBS=ON \
    && cmake --build build \
    && cmake --install build

# Finally copy the entrypoint script
# This is last to enable quick updates to the script after initial local build.
COPY ./docker/entrypoint.sh /entrypoint.sh


ENTRYPOINT ["/entrypoint.sh"]
