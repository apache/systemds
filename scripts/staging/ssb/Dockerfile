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

# Follow the tutorial: https://docs.docker.com/compose/gettingstarted/#step-1-set-up

FROM postgres:latest
# Init the data and load to the database with a sql script.
COPY other/ssb_init.sql /docker-entrypoint-initdb.d/

# Copy data into container
#COPY data_dir tmp

#WORKDIR /tmp
#RUN sed -i 's/|$//' "customer.tbl"
#RUN sed -i 's/|$//' "part.tbl"
#RUN sed -i 's/|$//' "supplier.tbl"
#RUN sed -i 's/|$//' "date.tbl"
#RUN sed -i 's/|$//' "lineorder.tbl"

