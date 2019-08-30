/*
 * Modifications Copyright 2019 Graz University of Technology
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
 
package org.tugraz.sysds.test.functions.mlcontext;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.tugraz.sysds.api.mlcontext.MLContext;
import org.tugraz.sysds.api.mlcontext.MLContextUtil;
import org.tugraz.sysds.test.AutomatedTestBase;

/**
 * Abstract class that can be used for MLContext tests.
 * <p>
 * Note that if using the setUp() method of MLContextTestBase, the test directory
 * and test name can be specified if needed in the subclass.
 * <p>
 * 
 * Example:
 * 
 * <pre>
 * public MLContextTestExample() {
 * 	testDir = this.getClass().getPackage().getName().replace(".", File.separator);
 * 	testName = this.getClass().getSimpleName();
 * }
 * </pre>
 *
 */
public abstract class MLContextTestBase extends AutomatedTestBase {

	protected static SparkSession spark;
	protected static JavaSparkContext sc;
	protected static MLContext ml;

	protected String testDir = null;
	protected String testName = null;

	@Override
	public void setUp() {
		Class<? extends MLContextTestBase> clazz = this.getClass();
		String dir = (testDir == null) ? "functions/mlcontext" : testDir;
		String name = (testName == null) ? clazz.getSimpleName() : testName;

		addTestConfiguration(dir, name);
		getAndLoadTestConfiguration(name);
	}

	@BeforeClass
	public static void setUpClass() {
		spark = createSystemDSSparkSession("SystemDS MLContext Test", "local");
		ml = new MLContext(spark);
		sc = MLContextUtil.getJavaSparkContext(ml);
	}

	@After
	public void tearDown() {
		super.tearDown();
	}

	@AfterClass
	public static void tearDownClass() {
		// stop underlying spark context to allow single jvm tests (otherwise
		// the next test that tries to create a SparkContext would fail)
		spark.stop();
		sc = null;
		spark = null;
		ml.close();
		ml = null;
	}
}
