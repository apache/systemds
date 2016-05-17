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
{% endcomment %}
-->

## General setup to run one of the Jupyter notebooks on IBM Bluemix:

* Clone the repository to download the notebooks
```
git clone https://github.com/apache/incubator-systemml.git
```

* Log on to https://console.ng.bluemix.net/ and create Apache Spark service:

![Setup screenshot](images/bluemix_screen.jpeg?raw=true "Setup screenshot")

* Go to Apache Spark service dashboard and click on notebook button:

![Setup screenshot](images/bluemix_spark_screen.jpeg?raw=true "Setup screenshot")

* Create a new notebook:

![Setup screenshot](images/bluemix_spark_screen2.jpeg?raw=true "Setup screenshot")

* Upload the notebook from this tutorial you want run on bluemix:

![Setup screenshot](images/bluemix_spark_screen3.jpeg?raw=true "Setup screenshot")

* Hurray, we now have a scala notebook running on bluemix:

![Setup screenshot](images/bluemix_spark_screen4.jpeg?raw=true "Setup screenshot")
