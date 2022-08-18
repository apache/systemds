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

package org.apache.sysds.test.functions.transform;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class TransformEncodeUDFTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "TransformEncodeUDF1"; //min-max
	private final static String TEST_NAME2 = "TransformEncodeUDF2"; //scale w/ defaults
    private final static String TEST_NAME3 = "TransformEncodeUDF3"; //simple custom UDF
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformEncodeUDFTest.class.getSimpleName() + "/";

	//dataset and transform tasks without missing values
	private final static String DATASET = "homes3/homes.csv";

	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}) );
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"R"}) );
	}

	@Test
	public void testUDF1Singlenode() {
		runTransformTest(ExecMode.SINGLE_NODE, TEST_NAME1);
	}

	@Test
	public void testUDF1Hybrid() {
		runTransformTest(ExecMode.HYBRID, TEST_NAME1);
	}

	@Test
	public void testUDF2Singlenode() {
		runTransformTest(ExecMode.SINGLE_NODE, TEST_NAME2);
	}

	@Test
	public void testUDF2Hybrid() {
        runTransformTest(ExecMode.HYBRID, TEST_NAME2);
	}

    @Test
    public void testUDF3Singlenode() {
        runTransformTest(ExecMode.SINGLE_NODE, TEST_NAME3);
    }

    @Test
    public void testUDF3Hybrid() {
        runTransformTest(ExecMode.HYBRID, TEST_NAME3);
    }

	private void runTransformTest(ExecMode rt, String testname)
	{
		//set runtime platform
		ExecMode rtold = setExecMode(rt);

		try
		{
			getAndLoadTestConfiguration(testname);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain",
				"-nvargs", "DATA=" + DATASET_DIR + DATASET, "R="+output("R")};

			//compare transformencode+scale vs transformencode w/ UDF
			runTest(true, false, null, -1);

			double ret = HDFSTool.readDoubleFromHDFSFile(output("R"));
			Assert.assertEquals(Double.valueOf(148*9), Double.valueOf(ret));

			if( rt == ExecMode.HYBRID ) {
				Long num = Long.valueOf(Statistics.getNoOfExecutedSPInst());
				Assert.assertEquals("Wrong number of executed Spark instructions: " + num, Long.valueOf(0), num);
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(rtold);
		}
	}
}
