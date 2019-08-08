/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.lineage;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class LineageRewriteTest extends AutomatedTestBase {
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "RewriteTest1";
	
	protected String TEST_CLASS_DIR = TEST_DIR + LineageRewriteTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 10;
	protected static final int numFeatures = 6;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
	}
	
	@Test
	public void testRewrite1() {
		testRewrite(TEST_NAME1);
	}
	
	private void testRewrite(String testname) {
		getAndLoadTestConfiguration(testname);
		List<String> proArgs = new ArrayList<String>();
		
		proArgs.add("-explain");
		proArgs.add("-lineage");
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(output("Res1"));
		proArgs.add(output("Res2"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		fullDMLScriptName = getScript();
		double[][] X = getRandomMatrix(numRecords, numFeatures, 0, 1, 0.8, -1);
		writeInputMatrixWithMTD("X", X, true);
		
		//run the test
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
	}
}
