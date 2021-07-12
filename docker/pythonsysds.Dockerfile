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
ENV MAVEN_VERSION 3.6.3
ENV MAVEN_HOME /usr/lib/mvn
ENV PATH $MAVEN_HOME/bin:$PATH
# Java
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

RUN apt-get update -qq \
	&& apt-get upgrade -y

RUN apt-get install -y --no-install-recommends \
		wget \
		git \
		ca-certificates \ 
	&& apt-get clean

# Maven and Java
RUN mkdir -p /usr/lib/jvm \
	&& wget -qO- \
https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u282-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u282b08.tar.gz | tar xzf - \
	&& mv jdk8u282-b08 /usr/lib/jvm/java-8-openjdk-amd64 \
	&& wget -qO- \
http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn

RUN git clone --depth 1 https://github.com/apache/systemds.git systemds

WORKDIR /usr/src/systemds/

RUN mvn clean package -P distribution
ENV SYSTEMDS_ROOT=/usr/src/systemds
ENV PATH $SYSTEMDS_ROOT/bin:$PATH

WORKDIR /usr/src/systemds/src/main/python

RUN apt-get install -y --no-install-recommends \
	python3 python3-pip && \
	apt-get clean && \
	python3 -m pip install --upgrade pip && \
	pip3 install numpy py4j wheel requests pandas && \
	python3 create_python_dist.py && \
	pip3 install .
	
ENV SYSDS_QUIET=1

WORKDIR /usr/src/systemds/

# Remove extra files.
RUN rm -r docker && \
	rm -r docs && \
	rm -r src && \
    rm -r /usr/lib/mvn && \
	rm -r .git && \
	rm -r CONTRIBUTING.md && \
	rm -r pom.xml && \ 
	rm -r ~/.m2

COPY docker/mountFolder/main.dml /input/main.dml

CMD ["python3", "/input/main.py"]
