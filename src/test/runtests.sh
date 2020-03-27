#/bin/bash
#-------------------------------------------------------------
#  Modifications Copyright 2020 Graz University of Technology
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
# under the License.
#-------------------------------------------------------------

<<COMMENT
This file runs all the java test classes.
The numbers on each line is the number of tests allowed to execute in parallel.
The tests are executed using:
    
    mvn surefire:test -Dtest=$class_name

Each java class will end up with a logging file in:

    target/surefire-reports/

COMMENT

build="$(mvn clean compile test-compile | grep 'BUILD')"
echo $build

grep_args="Tests run: \|R is finished\|ERROR"
resfile="temp/testres.log"

if [[ $build == *"SUCCESS"* ]]; then
	# Intensionally made into multiple lines such that
	# one is able to comment out specific tests manually.

	# Currently some tests fail because the matrix files
	# generated sometime overlap in naming.
	# TODO: make all tests not use the same scratchspace file.

	# Applications TOTAL ~6min
	mvn surefire:test -Dtest=org.tugraz.sysds.test.applications.** | grep $grep_args | tee -a $resfile

	# Component TOTAL ~13 sec
	mvn surefire:test -Dtest=org.tugraz.sysds.test.component.** 2>&1 | grep $grep_args | tee -a $resfile

	# Functions Total: ~ 1hour 10min
	
	# ~ 13 min
	mvn surefire:test -Dtest=org.tugraz.sysds.test.functions.a*.**,org.tugraz.sysds.test.functions.b*.**, 2>&1 | grep $grep_args | tee -a $resfile
	
	# ~ 9 min
	mvn surefire:test -Dtest=org.tugraz.sysds.test.functions.c*.** 2>&1 | grep $grep_args | tee -a $resfile
	
	# ~ ?? Does not end
	# TODO: Look into Data tests.
	# mvn surefire:test -Dtest=org.tugraz.sysds.test.functions.d*.** 2>&1 | grep $grep_args | tee -a $resfile
	
	# ~ 10 min
	mvn surefire:test -Dtest=org.tugraz.sysds.test.functions.f*.**,org.tugraz.sysds.test.functions.i*.**,org.tugraz.sysds.test.functions.j*.**,org.tugraz.sysds.test.functions.l*.**,org.tugraz.sysds.test.functions.m*.** 2>&1 | grep $grep_args | tee -a $resfile
	
	# ~ 19 min
	mvn surefire:test -Dtest=org.tugraz.sysds.test.functions.n*.**,org.tugraz.sysds.test.functions.par*.**,org.tugraz.sysds.test.functions.q*.**,org.tugraz.sysds.test.functions.r*.**,org.tugraz.sysds.test.functions.t*.**,org.tugraz.sysds.test.functions.u*.**,org.tugraz.sysds.test.functions.v*.** 2>&1 | grep $grep_args | tee -a $resfile

	# ~ 4 min
	# Large resoruce requirements:
	mvn surefire:test -Dtest=org.tugraz.sysds.test.functions.parameterserv* | grep $grep_args | tee -a $resfile


else
	echo "Compiling Failed"
fi
