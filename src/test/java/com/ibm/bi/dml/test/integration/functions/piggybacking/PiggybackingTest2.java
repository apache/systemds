/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.piggybacking;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class PiggybackingTest2 extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "Piggybacking_iqm";
	private final static String TEST_DIR = "functions/piggybacking/";

	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "x", "iqm.scalar" })   ); 
	}
	
	/**
	 * Tests for a piggybacking bug.
	 * 
	 * Specific issue is that the combineunary lop gets piggybacked
	 * into GMR while it should only be piggybacked into SortMR job.
	 */
	@Test
	public void testPiggybacking_iqm()
	{		

		RUNTIME_PLATFORM rtold = rtplatform;
		
		// bug can be reproduced only when exec mode is HADOOP 
		rtplatform = RUNTIME_PLATFORM.HADOOP;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + OUTPUT_DIR + config.getOutputFiles()[0] };

		loadTestConfiguration(config);
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
	
		HashMap<CellIndex, Double> d = TestUtils.readDMLScalarFromHDFS(HOME + OUTPUT_DIR + config.getOutputFiles()[0]);
		
		Assert.assertEquals(d.get(new CellIndex(1,1)), Double.valueOf(1.0), 1e-10);
		
		rtplatform = rtold;
	}

}