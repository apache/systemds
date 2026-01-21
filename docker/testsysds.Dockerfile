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
# Stage 1: Build SEAL, OpenBLAS, MKL
FROM ubuntu:noble@sha256:728785b59223d755e3e5c5af178fab1be7031f3522c5ccd7a0b32b80d8248123 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    tar \
    git \
    ca-certificates \
    gnupg \
    doxygen \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /seal

# Install SEAL
ARG SEAL_VERSION="3.7.0"
RUN wget -qO- https://github.com/microsoft/SEAL/archive/refs/tags/v${SEAL_VERSION}.tar.gz | tar xzf - \
    && cd SEAL-${SEAL_VERSION} \
    && cmake -S . -B build -DBUILD_SHARED_LIBS=ON \
    && cmake --build build \
    && cmake --install build --prefix /seal-install

WORKDIR /openBLAS

# Install OpenBLAS
ARG OPENBLAS_VERSION="0.3.26"
RUN wget -qO- https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v${OPENBLAS_VERSION}.tar.gz | tar xzf - \
    && cd OpenBLAS-${OPENBLAS_VERSION} \
    && make -j$(nproc) \
    && make install PREFIX=/openBLAS-install

WORKDIR /mkl

# Install MKL
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update && apt-get install -y --no-install-recommends intel-oneapi-mkl-devel

# Delete unused libraries, since MKL libraries take up a lot of disk space
RUN find /opt/intel/oneapi/mkl/2025.3/lib \( \ 
    -name '*ilp64*' -o \
    -name 'libmkl_gnu_thread*' -o \
    -name 'libmkl_tbb_thread*' -o \
    -name 'libmkl_sycl*' -o \
    -name '*_openmpi_*' -o \
    -name '*_intelmpi_*' -o \
    -name 'libmkl_vml*' -o \
    -name 'libmkl_scalapack*' \
    \) -delete

# Install MKL-DNN
ARG MKL_DNN_VERSION="0.21.5"
RUN wget -qO- https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v${MKL_DNN_VERSION}.tar.gz | tar xzf - \
    && cd oneDNN-${MKL_DNN_VERSION} \
    && mkdir -p build \
    && cd build \
    && cmake -DWITH_EXAMPLE=OFF -DWITH_TEST=OFF -DCMAKE_INSTALL_PREFIX=/mkldnn-install .. \
    && make \
    && make install 

# Stage 2: Final image with R, JDK, Maven, SEAL, OpenBLAS, MKL
FROM ubuntu:noble@sha256:728785b59223d755e3e5c5af178fab1be7031f3522c5ccd7a0b32b80d8248123 

WORKDIR /usr/src/
ENV MAVEN_VERSION=3.9.9
ENV MAVEN_HOME=/usr/lib/mvn

ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.15+6
ENV PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

RUN apt-get update && apt-get install -y locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8

RUN apt-get install -y --no-install-recommends \
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
    gfortran \
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

ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV LD_LIBRARY_PATH=/opt/hadoop/lib/native
ENV HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"

RUN mkdir -p $HADOOP_HOME/lib/native \
	&& wget -q https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar --strip-components=2 -xzf hadoop-${HADOOP_VERSION}.tar.gz \
        hadoop-${HADOOP_VERSION}/lib/native && \
    mv native/libhadoop.so.1.0.0 /opt/hadoop/lib/native && \
	mv native/libhadoop.so /opt/hadoop/lib/native && \
    rm hadoop-${HADOOP_VERSION}.tar.gz && \
	rm -rf native

# Copy SEAL
COPY --from=build /seal-install/lib/ /usr/local/lib/
COPY --from=build /seal-install/include/ /usr/local/include/

# Copy OpenBLAS
COPY --from=build /openBLAS-install/lib/ /usr/local/lib/
COPY --from=build /openBLAS-install/include/ /usr/local/include/

# Copy MKL
COPY --from=build /opt/intel/oneapi/mkl/2025.3/lib /usr/local/lib/
COPY --from=build /opt/intel/oneapi/mkl/2025.3/include /usr/local/include/
COPY --from=build /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so /usr/local/lib/
COPY --from=build /mkldnn-install/lib /usr/local/lib/
COPY --from=build /mkldnn-install/include /usr/local/include/

ENV LD_LIBRARY_PATH=/opt/hadoop/lib/native;/usr/local/lib/


# Finally copy the entrypoint script
# This is last to enable quick updates to the script after initial local build.
COPY ./docker/entrypoint.sh /entrypoint.sh


ENTRYPOINT ["/entrypoint.sh"]
