/*
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

package org.apache.sysds.test.functions.mlcontext;

import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.sysds.api.mlcontext.MLContext;
import org.apache.sysds.api.mlcontext.MLContextUtil;
import org.apache.sysds.api.mlcontext.MLResults;
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.api.mlcontext.ScriptExecutor;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;

/**
 * Abstract class that can be used for MLContext tests.
 * <p>
 * Note that if using the setUp() method of MLContextTestBase, the test directory and test name can be specified if
 * needed in the subclass.
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
	protected Level _oldLevel = null;
	protected boolean _enableTracing = true;
	
	@Override
	public void setUp() {
		Class<? extends MLContextTestBase> clazz = this.getClass();
		String dir = (testDir == null) ? "functions/mlcontext/" : testDir;
		String name = (testName == null) ? clazz.getSimpleName() : testName;

		addTestConfiguration(dir, name);
		getAndLoadTestConfiguration(name);
		
		//run all mlcontext tests in loglevel trace to improve test coverage
		//of all logging in various components
		if( _enableTracing ) {
			_oldLevel = Logger.getLogger("org.apache.sysds").getLevel();
			Logger.getLogger("org.apache.sysds").setLevel( Level.TRACE );
		}
	}
	
	@BeforeClass
	public static void setUpClass() {
		spark = createSystemDSSparkSession("SystemDS MLContext Test", "local");
		ml = new MLContext(spark);
		sc = MLContextUtil.getJavaSparkContext(ml);
	}

	@After
	@Override
	public void tearDown() {
		super.tearDown();
		if(_enableTracing)
			Logger.getLogger("org.apache.sysds").setLevel( _oldLevel );
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

	public static Pair<MLResults, String> executeAndCaptureStdOut(Script script){
		ByteArrayOutputStream buff = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(buff);
		PrintStream old = System.out;
		System.setOut(ps);
		MLResults res = safeExecute(buff, script, null);
		System.out.flush();
		System.setOut(old);

		return new ImmutablePair<>(res, buff.toString());
	}

	public static Pair<MLResults, String> executeAndCaptureStdOut(Script script, Class<?> expectedException){
		if(expectedException == null){
			return executeAndCaptureStdOut(script);
		}

		ByteArrayOutputStream buff = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(buff);
		PrintStream old = System.out;
		System.setOut(ps);
		MLResults res= unsafeExecute(script, null, expectedException);
		System.out.flush();
		System.setOut(old);

		return new ImmutablePair<>(res, buff.toString());
	}

	public static Pair<MLResults, String> executeAndCaptureStdOut(Script script, ScriptExecutor sce){
		ByteArrayOutputStream buff = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(buff);
		PrintStream old = System.out;
		System.setOut(ps);
		MLResults res = safeExecute(buff, script,sce);
		System.out.flush();
		System.setOut(old);

		return new ImmutablePair<>(res, buff.toString());
	}

	private static MLResults safeExecute(ByteArrayOutputStream buff, Script script, ScriptExecutor sce){
		try {
			MLResults res = sce == null ? ml.execute(script): ml.execute(script,sce);
			return res;
		}
		catch(Exception e) {
			StringBuilder errorMessage = new StringBuilder();
			errorMessage.append("\nfailed to run script: ");
			errorMessage.append("\nStandard Out:");
			errorMessage.append("\n" + buff);
			errorMessage.append("\nStackTrace:");
			errorMessage.append(AutomatedTestBase.getStackTraceString(e, 0));
			fail(errorMessage.toString());
		}
		return null;
	}

	private static MLResults unsafeExecute(Script script, ScriptExecutor sce, Class<?> expectedException){
		try {

			MLResults res = sce == null ? ml.execute(script): ml.execute(script, sce);
			return res;
		}
		catch(Exception e) {
			if(!(e.getClass().equals(expectedException))){

				StringBuilder errorMessage = new StringBuilder();
				errorMessage.append("\nfailed to run script: ");
				errorMessage.append("\nStackTrace:");
				errorMessage.append(AutomatedTestBase.getStackTraceString(e, 0));
				fail(errorMessage.toString());
			}
		}
		return null;
	}
}
