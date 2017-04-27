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

package org.apache.sysml.test.integration.scripts.nn;

import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;

/**
 * Test the SystemML deep learning library, `nn`.
 */
public class NNTest extends AutomatedTestBase {

	private static final String TEST_NAME = "NNTest";
	private static final String TEST_DIR = "scripts/";
	private static final String TEST_SCRIPT = "scripts/nn/test/run_tests.dml";
	private static final String ERROR_STRING = "ERROR:";

	private static SparkSession spark;
	private static MLContext ml;

	@BeforeClass
	public static void setUpClass() {
		spark = createSystemMLSparkSession("MLContextTest", "local");
		ml = new MLContext(spark);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testNNLibrary() {
		Script script = dmlFromFile(TEST_SCRIPT);
		setUnexpectedStdOut(ERROR_STRING);
		ml.execute(script);
	}

	@After
	public void tearDown() {
		super.tearDown();
	}

	@AfterClass
	public static void tearDownClass() {
		// stop underlying spark context to allow single jvm tests (otherwise the
		// next test that tries to create a SparkContext would fail)
		spark.stop();
		spark = null;

		// clear status mlcontext and spark exec context
		ml.close();
		ml = null;
	}
}
