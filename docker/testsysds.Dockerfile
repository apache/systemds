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

FROM ubuntu:24.04@sha256:6015f66923d7afbc53558d7ccffd325d43b4e249f41a6e93eef074c9505d2233

WORKDIR /usr/src/
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn

ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/

RUN apt-get update -qq \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
		libcurl4-openssl-dev \
		libxml2-dev \
		locales \
		software-properties-common \
		dirmngr \
		gnupg \
		apt-transport-https \
		wget \
		ca-certificates \
		git \
		cmake \
		patchelf \
	&& apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
	&& add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
	&& apt-get update -qq \
	&& apt-get upgrade -y \
	&& echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8 \
	&& mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jdk_x64_linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6 /usr/lib/jvm/jdk-17.0.15+6 \
	&& wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn

# Install R Base
RUN apt-get install -y --no-install-recommends \
		libssl-dev \
		r-base \
		r-base-dev \
		r-base-core


# Install R packages
COPY ./src/test/scripts/installDependencies.R installDependencies.R		
RUN Rscript installDependencies.R \
	&& rm -rf installDependencies.R \
	&& rm -rf /var/lib/apt/lists/*

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
