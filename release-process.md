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


## Release Documentation

The `SYSTEMML_VERSION` value in docs/_config.yml should be updated to the correct release version. The documentation
site should be built.
The SystemML documentation site should be deployed to a docs version folder within the main website project (using
svn). As an example, the documentation site for SystemML version 0.11.0 should be available
at http://systemml.apache.org/docs/0.11.0.

The Javadocs should be generated for the project and should be deployed to a docs version folder, such as
http://systemml.apache.org/docs/0.11.0/api/java. Any other docs, such as Scaladocs if they are available, should
be deployed to corresponding locations. Note that the version number specified in the Javadocs is determined by the project
version number in the project pom.xml file.

Additionally, the Javadocs should be deployed to http://systemml.apache.org/docs/latest/api/java
if the Javadocs have not already been deployed to this location.


# Release Candidate Checklist

## All Artifacts and Checksums Present

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that each expected artifact is present at [https://dist.apache.org/repos/dist/dev/incubator/systemml/](https://dist.apache.org/repos/dist/dev/incubator/systemml/) and that each artifact has accompanying
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

	$ git clone https://github.com/apache/incubator-systemml.git
	$ cd incubator-systemml
	$ git tag -l
	$ git checkout tags/0.10.0-incubating-rc1 -b 0.10.0-incubating-rc1
	$ mvn -Dmaven.repo.local=$HOME/.m2/temp-repo clean package -P distribution


## Test Suite Passes

<a href="#release-candidate-checklist">Up to Checklist</a>

The entire test suite should pass on Windows, OS X, and Linux.
The test suite can be run using:

	$ mvn clean verify


## All Binaries Execute

<a href="#release-candidate-checklist">Up to Checklist</a>

Validate that all of the binary artifacts can execute, including those artifacts packaged
in other artifacts (in the tar.gz and zip artifacts).

The build artifacts should be downloaded from [https://dist.apache.org/repos/dist/dev/incubator/systemml/](https://dist.apache.org/repos/dist/dev/incubator/systemml/) and these artifacts should be tested, as in
this OS X example.

	# download artifacts
	wget -r -nH -nd -np -R index.html* https://dist.apache.org/repos/dist/dev/incubator/systemml/0.10.0-incubating-rc1/

	# verify standalone tar.gz works
	tar -xvzf systemml-0.10.0-incubating-standalone.tar.gz
	cd systemml-0.10.0-incubating-standalone
	echo "print('hello world');" > hello.dml
	./runStandaloneSystemML.sh hello.dml
	cd ..

	# verify main jar works
	mkdir lib
	cp -R systemml-0.10.0-incubating-standalone/lib/* lib/
	rm lib/systemml-0.10.0-incubating.jar
	java -cp ./lib/*:systemml-0.10.0-incubating.jar org.apache.sysml.api.DMLScript -s "print('hello world');"

	# verify standalone jar works
	java -jar systemml-0.10.0-incubating-standalone.jar -s "print('hello world');"

	# verify src works
	tar -xvzf systemml-0.10.0-incubating-src.tar.gz
	cd systemml-0.10.0-incubating-src
	mvn clean package -P distribution
	cd target/
	java -cp ./lib/*:systemml-0.10.0-incubating.jar org.apache.sysml.api.DMLScript -s "print('hello world');"
	java -cp ./lib/*:SystemML.jar org.apache.sysml.api.DMLScript -s "print('hello world');"
	java -jar systemml-0.10.0-incubating-standalone.jar -s "print('hello world');"
	cd ..
	cd ..

	# verify in-memory jar works
	echo "import org.apache.sysml.api.jmlc.*;public class JMLCEx {public static void main(String[] args) throws Exception {Connection conn = new Connection();PreparedScript script = conn.prepareScript(\"print('hello world');\", new String[]{}, new String[]{}, false);script.executeScript();}}" > JMLCEx.java
	javac -cp systemml-0.10.0-incubating-inmemory.jar JMLCEx.java
	java -cp .:systemml-0.10.0-incubating-inmemory.jar JMLCEx

	# verify distrib tar.gz works
	tar -xvzf systemml-0.10.0-incubating.tar.gz
	cd systemml-0.10.0-incubating
	java -cp ../lib/*:SystemML.jar org.apache.sysml.api.DMLScript -s "print('hello world');"

	# verify spark batch mode
	export SPARK_HOME=/Users/deroneriksson/spark-1.5.1-bin-hadoop2.6
	$SPARK_HOME/bin/spark-submit SystemML.jar -s "print('hello world');" -exec hybrid_spark

	# verify hadoop batch mode
	hadoop jar SystemML.jar -s "print('hello world');"


Here is an example of doing a basic
sanity check on OS X after building the artifacts manually.

	# build distribution artifacts
	mvn clean package -P distribution

	cd target

	# verify main jar works
	java -cp ./lib/*:systemml-0.10.0-incubating.jar org.apache.sysml.api.DMLScript -s "print('hello world');"

	# verify SystemML.jar works
	java -cp ./lib/*:SystemML.jar org.apache.sysml.api.DMLScript -s "print('hello world');"

	# verify standalone jar works
	java -jar systemml-0.10.0-incubating-standalone.jar -s "print('hello world');"

	# verify src works
	tar -xvzf systemml-0.10.0-incubating-src.tar.gz
	cd systemml-0.10.0-incubating-src
	mvn clean package -P distribution
	cd target/
	java -cp ./lib/*:systemml-0.10.0-incubating.jar org.apache.sysml.api.DMLScript -s "print('hello world');"
	java -cp ./lib/*:SystemML.jar org.apache.sysml.api.DMLScript -s "print('hello world');"
	java -jar systemml-0.10.0-incubating-standalone.jar -s "print('hello world');"
	cd ..
	cd ..

	# verify in-memory jar works
	echo "import org.apache.sysml.api.jmlc.*;public class JMLCEx {public static void main(String[] args) throws Exception {Connection conn = new Connection();PreparedScript script = conn.prepareScript(\"print('hello world');\", new String[]{}, new String[]{}, false);script.executeScript();}}" > JMLCEx.java
	javac -cp systemml-0.10.0-incubating-inmemory.jar JMLCEx.java
	java -cp .:systemml-0.10.0-incubating-inmemory.jar JMLCEx

	# verify standalone tar.gz works
	tar -xvzf systemml-0.10.0-incubating-standalone.tar.gz
	cd systemml-0.10.0-incubating-standalone
	echo "print('hello world');" > hello.dml
	./runStandaloneSystemML.sh hello.dml
	cd ..

	# verify distrib tar.gz works
	tar -xvzf systemml-0.10.0-incubating.tar.gz
	cd systemml-0.10.0-incubating
	java -cp ../lib/*:SystemML.jar org.apache.sysml.api.DMLScript -s "print('hello world');"

	# verify spark batch mode
	export SPARK_HOME=/Users/deroneriksson/spark-1.5.1-bin-hadoop2.6
	$SPARK_HOME/bin/spark-submit SystemML.jar -s "print('hello world');" -exec hybrid_spark

	# verify hadoop batch mode
	hadoop jar SystemML.jar -s "print('hello world');"


## Check LICENSE and NOTICE Files

<a href="#release-candidate-checklist">Up to Checklist</a>

Each artifact *must* contain LICENSE and NOTICE files. These files must reflect the
contents of the artifacts. If the project dependencies (ie, libraries) have changed
since the last release, the LICENSE and NOTICE files must be updated to reflect these
changes.

Each artifact *should* contain a DISCLAIMER file.

For more information, see:

1. <http://incubator.apache.org/guides/releasemanagement.html>
2. <http://www.apache.org/dev/licensing-howto.html>


## Src Artifact Builds and Tests Pass

<a href="#release-candidate-checklist">Up to Checklist</a>

The project should be built using the `src` (tar.gz and zip) artifacts.
In addition, the test suite should be run using an `src` artifact and
the tests should pass.

	tar -xvzf systemml-0.10.0-incubating-src.tar.gz
	cd systemml-0.10.0-incubating-src
	mvn clean package -P distribution
	mvn verify


## Single-Node Standalone

<a href="#release-candidate-checklist">Up to Checklist</a>

The standalone tar.gz and zip artifacts contain `runStandaloneSystemML.sh` and `runStandaloneSystemML.bat`
files. Verify that one or more algorithms can be run on a single node using these
standalone distributions.

Here is an example based on the [Standalone Guide](http://apache.github.io/incubator-systemml/standalone-guide.html)
demonstrating the execution of an algorithm (on OS X).

	$ tar -xvzf systemml-0.10.0-incubating-standalone.tar.gz
	$ cd systemml-0.10.0-incubating-standalone
	$ wget -P data/ http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
	$ echo '{"rows": 306, "cols": 4, "format": "csv"}' > data/haberman.data.mtd
	$ echo '1,1,1,2' > data/types.csv
	$ echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd
	$ ./runStandaloneSystemML.sh scripts/algorithms/Univar-Stats.dml -nvargs X=data/haberman.data TYPES=data/types.csv STATS=data/univarOut.mtx CONSOLE_OUTPUT=TRUE


## Single-Node Spark

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that SystemML runs algorithms on Spark locally.

Here is an example of running the `Univar-Stats.dml` algorithm on random generated data.

	$ tar -xvzf systemml-0.10.0-incubating.tar.gz
	$ cd systemml-0.10.0-incubating
	$ export SPARK_HOME=/Users/deroneriksson/spark-1.5.1-bin-hadoop2.6
	$ $SPARK_HOME/bin/spark-submit SystemML.jar -f scripts/datagen/genRandData4Univariate.dml -exec hybrid_spark -args 1000000 100 10 1 2 3 4 uni.mtx
	$ echo '1' > uni-types.csv
	$ echo '{"rows": 1, "cols": 1, "format": "csv"}' > uni-types.csv.mtd
	$ $SPARK_HOME/bin/spark-submit SystemML.jar -f scripts/algorithms/Univar-Stats.dml -exec hybrid_spark -nvargs X=uni.mtx TYPES=uni-types.csv STATS=uni-stats.txt CONSOLE_OUTPUT=TRUE


## Single-Node Hadoop

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that SystemML runs algorithms on Hadoop locally.

Based on the "Single-Node Spark" setup above, the `Univar-Stats.dml` algorithm could be run as follows:

	$ hadoop jar SystemML.jar -f scripts/algorithms/Univar-Stats.dml -nvargs X=uni.mtx TYPES=uni-types.csv STATS=uni-stats.txt CONSOLE_OUTPUT=TRUE


## Notebooks

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that SystemML can be executed from Jupyter and Zeppelin notebooks.
For examples, see the [Spark MLContext Programming Guide](http://apache.github.io/incubator-systemml/spark-mlcontext-programming-guide.html).


## Performance Suite

<a href="#release-candidate-checklist">Up to Checklist</a>

Verify that the performance suite located at scripts/perftest/ executes on Spark and Hadoop. Testing should
include 80MB, 800MB, 8GB, and 80GB data sizes.


# Voting

Following a successful release candidate vote by SystemML PMC members on the SystemML mailing list, the release candidate
is voted on by Incubator PMC members on the general incubator mailing list. If this vote succeeds, the release candidate
has been approved.


# Release


## Release Deployment

To be written. (What steps need to be done? How is the release deployed to the central maven repo? What updates need to
happen to the main website, such as updating the Downloads page? Where do the release notes for the release go?)


