<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# Learned Sample Selection

## Data Preparation

    mkdir -p data;
    chmod 755 data;
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -o data/Adult.csv;
    sed -i '$d' data/Adult.csv; # fix empty line at end of file

## Run Individual Scripts

    java -Xmx4g -Xms4g -cp ./lib/*:./SystemDS.jar org.apache.sysds.api.DMLScript \
        -f XYZ.dml -debug -stats -explain

