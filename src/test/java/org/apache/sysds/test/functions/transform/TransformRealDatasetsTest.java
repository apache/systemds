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

public class TransformRealDatasetsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformPopByCitizenship";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformCSVFrameEncodeReadTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET = "popbycitizenship.csv";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}
	
	@Test
	public void testPopByCitizenshipCP() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testPopByCitizenshipSpark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK);
	}
	
	private void runTransformTest(String testname, ExecMode rt)
	{
		//set runtime platform
		ExecMode rtold = setExecMode(rt);
		
		try
		{
			getAndLoadTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats","-args",  DATASET_DIR + DATASET, output("R")};
			
			runTest(null);
			
			double R2 = HDFSTool.readDoubleFromHDFSFile(output("R"));
			Assert.assertTrue(R2 > 0.8);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(rtold);
		}
	}
}
