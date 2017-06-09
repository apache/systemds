---
layout: global
title: SystemML Release Process
description: Description of the SystemML release process and validation.
displayTitle: SystemML Release Process
---
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

* This will become a table of contents (this text will be scraped).
{:toc}


# Release Candidate Build and Deployment

To be written. (Describe how the release candidate is built, including checksums. Describe how
the release candidate is deployed to servers for review.)


# Release Candidate Checklist

## All Artifacts and Checksums Present

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that each expected artifact is present at [https://dist.apache.org/repos/dist/dev/systemml/](https://dist.apache.org/repos/dist/dev/systemml/) and that each artifact has accompanying
checksums (such as .asc and .md5).


## Release Candidate Build

<a href="#release-candidate-checklist">Up to Checklist</a>

The release candidate should build on Windows, OS X, and Linux. To do this cleanly,
the following procedure can be performed.

Clone the Apache SystemML GitHub repository
to an empty location. Next, check out the release tag. Following
this, build the distributions using Maven. This should be performed
with an empty local Maven repository.

Here is an example:

	$ git clone https://github.com/apache/systemml.git
	$ cd systemml
	$ git tag -l
	$ git checkout tags/1.0.0-rc1 -b 1.0.0-rc1
	$ mvn -Dmaven.repo.local=$HOME/.m2/temp-repo clean package -P distribution


## Test Suite Passes

<a href="#release-candidate-checklist">Up to Checklist</a>

The entire test suite should pass on Windows, OS X, and Linux.
The test suite can be run using:

	$ mvn clean verify


## All Binaries Execute

<a href="#release-candidate-checklist">Up to Checklist</a>

Validate that all of the binary artifacts can execute, including those artifacts packaged
in other artifacts (in the tgz and zip artifacts).

The build artifacts should be downloaded from [https://dist.apache.org/repos/dist/dev/systemml/](https://dist.apache.org/repos/dist/dev/systemml/) and these artifacts should be tested, as in
this OS X example.

	# download artifacts
	wget -r -nH -nd -np -R 'index.html*' https://dist.apache.org/repos/dist/dev/systemml/1.0.0-rc1/

	# verify standalone tgz works
	tar -xvzf systemml-1.0.0-bin.tgz
	cd systemml-1.0.0-bin
	echo "print('hello world');" > hello.dml
	./runStandaloneSystemML.sh hello.dml
	cd ..

	# verify standalon zip works
	rm -rf systemml-1.0.0-bin
	unzip systemml-1.0.0-bin.zip
	cd systemml-1.0.0-bin
	echo "print('hello world');" > hello.dml
	./runStandaloneSystemML.sh hello.dml
	cd ..

	# verify src works
	tar -xvzf systemml-1.0.0-src.tgz
	cd systemml-1.0.0-src
	mvn clean package -P distribution
	cd target/
	java -cp "./lib/*:systemml-1.0.0.jar" org.apache.sysml.api.DMLScript -s "print('hello world');"
	java -cp "./lib/*:SystemML.jar" org.apache.sysml.api.DMLScript -s "print('hello world');"
	cd ../..

	# verify spark batch mode
	export SPARK_HOME=~/spark-2.1.0-bin-hadoop2.7
	cd systemml-1.0.0-bin/target/lib
	$SPARK_HOME/bin/spark-submit systemml-1.0.0.jar -s "print('hello world');" -exec hybrid_spark

	# verify hadoop batch mode
	hadoop jar systemml-1.0.0.jar -s "print('hello world');"


	# verify python artifact
	# install numpy, pandas, scipy & set SPARK_HOME
	pip install numpy
	pip install pandas
	pip install scipy
	export SPARK_HOME=~/spark-2.1.0-bin-hadoop2.7
	# get into the pyspark prompt
	cd systemml-1.0.0
	$SPARK_HOME/bin/pyspark --driver-class-path systemml-java/systemml-1.0.0.jar
	# Use this program at the prompt:
	import systemml as sml
	import numpy as np
	m1 = sml.matrix(np.ones((3,3)) + 2)
	m2 = sml.matrix(np.ones((3,3)) + 3)
	m2 = m1 * (m2 + m1)
	m4 = 1.0 - m2
	m4.sum(axis=1).toNumPy()

	# This should be printed
	# array([[-60.],
	#       [-60.],
	#       [-60.]])



## Python Tests

For Spark 1.*, the Python tests at (`src/main/python/tests`) can be executed in the following manner:

	PYSPARK_PYTHON=python3 pyspark --driver-class-path SystemML.jar test_matrix_agg_fn.py
	PYSPARK_PYTHON=python3 pyspark --driver-class-path SystemML.jar test_matrix_binary_op.py
	PYSPARK_PYTHON=python3 pyspark --driver-class-path SystemML.jar test_mlcontext.py
	PYSPARK_PYTHON=python3 pyspark --driver-class-path SystemML.jar test_mllearn_df.py
	PYSPARK_PYTHON=python3 pyspark --driver-class-path SystemML.jar test_mllearn_numpy.py

For Spark 2.*, pyspark can't be used to run the Python tests, so they can be executed using
spark-submit:

	spark-submit --driver-class-path SystemML.jar test_matrix_agg_fn.py
	spark-submit --driver-class-path SystemML.jar test_matrix_binary_op.py
	spark-submit --driver-class-path SystemML.jar test_mlcontext.py
	spark-submit --driver-class-path SystemML.jar test_mllearn_df.py
	spark-submit --driver-class-path SystemML.jar test_mllearn_numpy.py


## Check LICENSE and NOTICE Files

<a href="#release-candidate-checklist">Up to Checklist</a>

Each artifact *must* contain LICENSE and NOTICE files. These files must reflect the
contents of the artifacts. If the project dependencies (ie, libraries) have changed
since the last release, the LICENSE and NOTICE files must be updated to reflect these
changes.

For more information, see:

1. <http://www.apache.org/dev/#releases>
2. <http://www.apache.org/dev/licensing-howto.html>


## Src Artifact Builds and Tests Pass

<a href="#release-candidate-checklist">Up to Checklist</a>

The project should be built using the `src` (tgz and zip) artifacts.
In addition, the test suite should be run using an `src` artifact and
the tests should pass.

	tar -xvzf systemml-1.0.0-src.tgz
	cd systemml-1.0.0-src
	mvn clean package -P distribution
	mvn verify


## Single-Node Standalone

<a href="#release-candidate-checklist">Up to Checklist</a>

The standalone tgz and zip artifacts contain `runStandaloneSystemML.sh` and `runStandaloneSystemML.bat`
files. Verify that one or more algorithms can be run on a single node using these
standalone distributions.

Here is an example based on the [Standalone Guide](http://apache.github.io/systemml/standalone-guide.html)
demonstrating the execution of an algorithm (on OS X).

	tar -xvzf systemml-1.0.0-bin.tgz
	cd systemml-1.0.0-bin
	wget -P data/ http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
	echo '{"rows": 306, "cols": 4, "format": "csv"}' > data/haberman.data.mtd
	echo '1,1,1,2' > data/types.csv
	echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd
	./runStandaloneSystemML.sh scripts/algorithms/Univar-Stats.dml -nvargs X=data/haberman.data TYPES=data/types.csv STATS=data/univarOut.mtx CONSOLE_OUTPUT=TRUE
	cd ..


## Single-Node Spark

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that SystemML runs algorithms on Spark locally.

Here is an example of running the `Univar-Stats.dml` algorithm on random generated data.

	cd systemml-1.0.0-bin/lib
	export SPARK_HOME=~/spark-2.1.0-bin-hadoop2.7
	$SPARK_HOME/bin/spark-submit systemml-1.0.0.jar -f ../scripts/datagen/genRandData4Univariate.dml -exec hybrid_spark -args 1000000 100 10 1 2 3 4 uni.mtx
	echo '1' > uni-types.csv
	echo '{"rows": 1, "cols": 1, "format": "csv"}' > uni-types.csv.mtd
	$SPARK_HOME/bin/spark-submit systemml-1.0.0.jar -f ../scripts/algorithms/Univar-Stats.dml -exec hybrid_spark -nvargs X=uni.mtx TYPES=uni-types.csv STATS=uni-stats.txt CONSOLE_OUTPUT=TRUE
	cd ..


## Single-Node Hadoop

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that SystemML runs algorithms on Hadoop locally.

Based on the "Single-Node Spark" setup above, the `Univar-Stats.dml` algorithm could be run as follows:

	cd systemml-1.0.0-bin/lib
	hadoop jar systemml-1.0.0.jar -f ../scripts/algorithms/Univar-Stats.dml -nvargs X=uni.mtx TYPES=uni-types.csv STATS=uni-stats.txt CONSOLE_OUTPUT=TRUE


## Notebooks

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that SystemML can be executed from Jupyter and Zeppelin notebooks.
For examples, see the [Spark MLContext Programming Guide](http://apache.github.io/systemml/spark-mlcontext-programming-guide.html).


## Performance Suite

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that the performance suite located at scripts/perftest/ executes on Spark and Hadoop. Testing should
include 80MB, 800MB, 8GB, and 80GB data sizes.


# Voting

Following a successful release candidate vote by SystemML PMC members on the SystemML mailing list, the release candidate
has been approved.


# Release


## Release Deployment

To be written. (What steps need to be done? How is the release deployed to Apache dist and the central maven repo?
Where do the release notes for the release go?)


## Documentation Deployment

This section describes how to deploy versioned project documentation to the main website.
Note that versioned project documentation is committed directly to the `svn` project's `docs` folder.
The versioned project documentation is not committed to the website's `git` project.

Checkout branch in main project (`systemml`).

	$ git checkout branch-1.0.0

In `systemml/docs/_config.yml`, set:

* `SYSTEMML_VERSION` to project version (1.0.0)
* `FEEDBACK_LINKS` to `false` (only have feedback links on `LATEST` docs)
* `API_DOCS_MENU` to `true` (adds `API Docs` menu to get to project javadocs)

Generate `docs/_site` by running `bundle exec jekyll serve` in `systemml/docs`.

	$ bundle exec jekyll serve

Verify documentation site looks correct.

In website `svn` project, create `systemml-website-site/docs/1.0.0` folder.

Copy contents of `systemml/docs/_site` to `systemml-website-site/docs/1.0.0`.

Delete any unnecessary files (`Gemfile`, `Gemfile.lock`).

Create `systemml-website-site/docs/1.0.0/api/java` folder for javadocs.

Update `systemml/pom.xml` project version to what should be displayed in javadocs (such as `1.0.0`).

Build project (which generates javadocs).

	$ mvn clean package -P distribution

Copy contents of `systemml/target/apidocs` to `systemml-website-site/docs/1.0.0/api/java`.

Open up `file:///.../systemml-website-site/docs/1.0.0/index.html` and verify `API Docs` &rarr; `Javadoc` link works and that the correct Javadoc version is displayed. Verify feedback links under `Issues` menu are not present.

Clean up any unnecessary files (such as deleting `.DS_Store` files on OS X).

	$ find . -name '.DS_Store' -type f -delete

Commit the versioned project documentation to `svn`:

	$ svn status
	$ svn add docs/1.0.0
	$ svn commit -m "Add 1.0.0 docs to website"

Update `systemml-website/_src/documentation.html` to include 1.0.0 link.

Start main website site by running `gulp` in `systemml-website`:

	$ gulp

Commit and push the update to `git` project.

	$ git add -u
	$ git commit -m "Add 1.0.0 link to documentation page"
	$ git push
	$ git push apache master

Copy contents of `systemml-website/_site` (generated by `gulp`) to `systemml-website-site`.
After doing so, we should see that `systemml-website-site/documentation.html` has been updated.

	$ svn status
	$ svn diff

Commit the update to `documentation.html` to publish the website update.

	$ svn commit -m "Add 1.0.0 link to documentation page"

The versioned project documentation is now deployed to the main website, and the
[Documentation Page](http://systemml.apache.org/documentation) contains a link to the versioned documentation.

