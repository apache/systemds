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

FROM python:3.12-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS compile-image

WORKDIR /usr/src/

RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
	wget \
	git \
	tar \
	ca-certificates \
	&& apt-get clean

# Maven
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn
ENV PATH=$MAVEN_HOME/bin:$PATH
# Java
ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH
ENV SYSTEMDS_ROOT=/usr/src/systemds
ENV PATH=$SYSTEMDS_ROOT/bin:$PATH
ENV SYSDS_QUIET=1

# Download JAVA
RUN mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jdk_x64_linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6 $JAVA_HOME

# Download Mvn 
RUN wget -qO- \
	http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - \ 
	&& mv apache-maven-$MAVEN_VERSION /usr/lib/mvn

RUN git clone --depth 1 https://github.com/apache/systemds.git systemds && \
	cd /usr/src/systemds/ && \
	mvn --no-transfer-progress clean package -P distribution

RUN cd /usr/src/systemds/src/main/python \
	apt-get install -y --no-install-recommends \
	python3 python3-pip && \
	apt-get clean && \
	python3 -m pip install --upgrade pip \
	&& pip3 install setuptools numpy py4j wheel requests pandas \
	&& python3 create_python_dist.py \
	&& pip3 install . \
	&& cd /usr/src/systemds/

COPY docker/mountFolder/main.py /input/main.py

FROM python:3.12-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends \
	wget \
	git \
	tar \
	ca-certificates \
	&& apt-get clean

# Java
ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$PATH
ENV SYSTEMDS_ROOT=/systemds
ENV PATH=$SYSTEMDS_ROOT/bin:$PATH
ENV SYSDS_QUIET=1

RUN mkdir -p /usr/lib/jvm \
	&& wget -qO- \
	https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.15%2B6/OpenJDK17U-jre_x64_linux_hotspot_17.0.15_6.tar.gz  | tar xzf - \
	&& mv jdk-17.0.15+6-jre $JAVA_HOME

COPY --from=compile-image /usr/src/systemds /systemds
COPY --from=compile-image /input/main.py /input/main.py

RUN cd /systemds/src/main/python \
	apt-get install -y --no-install-recommends \
	python3 python3-pip && \
	apt-get clean && \
	python3 -m pip install --upgrade pip \
	&& pip3 install setuptools numpy py4j wheel requests pandas \
	&& python3 create_python_dist.py \
	&& pip3 install . \
	&& cd /systemds/

WORKDIR /input

CMD ["python3", "/input/main.py"]
