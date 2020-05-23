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

# Use Alpine OpenJDK 8 base
FROM openjdk:8-alpine

WORKDIR /usr/src/

# Install Maven
# Credit https://github.com/Zenika/alpine-maven/blob/master/jdk8/Dockerfile

ENV MAVEN_VERSION 3.6.3
ENV MAVEN_HOME /usr/lib/mvn
ENV PATH $MAVEN_HOME/bin:$PATH

RUN wget http://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz && \
	tar -zxvf apache-maven-$MAVEN_VERSION-bin.tar.gz && \
	rm apache-maven-$MAVEN_VERSION-bin.tar.gz && \
	mv apache-maven-$MAVEN_VERSION /usr/lib/mvn

# Install Extras
RUN apk add --no-cache git bash

RUN git clone https://github.com/apache/systemml.git systemds

WORKDIR /usr/src/systemds/

RUN mvn clean package -P distribution

# Remove Maven since it is not needed for running the system
RUN rm -r /usr/lib/mvn

ENV SYSTEMDS_ROOT=/usr/src/systemds
ENV PATH $SYSTEMDS_ROOT/bin:$PATH

# Remove extra files.
RUN rm -r src/ && \
	rm -r .git

COPY docker/mountFolder/main.dml /input/main.dml

CMD ["systemds", "/input/main.dml"]
