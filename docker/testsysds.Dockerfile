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
# Stage 1: Build SEAL
FROM debian:bullseye-slim@sha256:b5f9bc44bdfbd9d551dfdd432607cbc6bb5d9d6dea726a1191797d7749166973 AS seal-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    tar \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /seal

# Install SEAL
RUN wget -qO- https://github.com/microsoft/SEAL/archive/refs/tags/v3.7.0.tar.gz | tar xzf - \
    && cd SEAL-3.7.0 \
    && cmake -S . -B build -DBUILD_SHARED_LIBS=ON \
    && cmake --build build \
    && cmake --install build --prefix /seal-install

# Stage 2: Final image with R, JDK, Maven, SEAL
FROM debian:bullseye-slim@sha256:b5f9bc44bdfbd9d551dfdd432607cbc6bb5d9d6dea726a1191797d7749166973

WORKDIR /usr/src/
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn

ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
	wget \
    cmake \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    ca-certificates \
    patchelf \
    git \
    libssl-dev \
	r-base-dev \
	r-base-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
	&& mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jdk_x64_linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6 $JAVA_HOME \
	&& wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn


# Install R packages
COPY ./src/test/scripts/installDependencies.R installDependencies.R
RUN Rscript installDependencies.R \
    && rm -f installDependencies.R

# Copy SEAL
COPY --from=seal-build /seal-install /usr/local

# Finally copy the entrypoint script
# This is last to enable quick updates to the script after initial local build.
COPY ./docker/entrypoint.sh /entrypoint.sh


ENTRYPOINT ["/entrypoint.sh"]
