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

# Install Maven
# Credit https://github.com/Zenika/alpine-maven/blob/master/jdk8/Dockerfile
# InstallR Guide: https://cran.r-project.org/

WORKDIR /usr/src/
ENV MAVEN_VERSION 3.6.3
ENV MAVEN_HOME /usr/lib/mvn
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

COPY ./src/test/scripts/installDependencies.R installDependencies.R
COPY ./docker/entrypoint.sh /entrypoint.sh

RUN apt-get update -qq \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
		libcurl4-openssl-dev \
		libxml2-dev \
		software-properties-common \
		dirmngr \
		gnupg \
		apt-transport-https \
		wget \
		ca-certificates \
	&& apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
	&& add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
	&& apt-get update -qq \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
		r-base \ 
		r-base-dev \
	&& Rscript installDependencies.R \
	&& rm -rf installDependencies.R \
	&& rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/lib/jvm \
	&& wget -qO- \
https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u282-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u282b08.tar.gz | tar xzf - \
	&& mv jdk8u282-b08 /usr/lib/jvm/java-8-openjdk-amd64 \
	&& wget -qO- \
http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
    && mv apache-maven-$MAVEN_VERSION /usr/lib/mvn


ENTRYPOINT ["/entrypoint.sh"]
