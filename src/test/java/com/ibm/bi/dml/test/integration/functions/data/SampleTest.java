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

package com.ibm.bi.dml.test.integration.functions.data;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 * Tests if Rand produces the same output, for a given set of parameters, across different (CP vs. MR) runtime platforms.   
 * 
 */
@RunWith(value = Parameterized.class)
public class SampleTest extends AutomatedTestBase 
{

	
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_NAME = "Sample";
	
	private enum TEST_TYPE { FOUR_INPUTS, THREE_INPUTS1, THREE_INPUTS2, TWO_INPUTS, ERROR };
	
	private TEST_TYPE test_type;
	private final static long RANGE=5000, SIZE=100, SIZE2=RANGE+10;
	
	private long _range, _size, _seed;
	private boolean _replace;
	
	public SampleTest(TEST_TYPE tp, long r, long s, boolean rep, long seed) {
		test_type = tp;
		_range = r;
		_size = s;
		_replace = rep;
		_seed = seed;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { 
				// 4 inputs
				{TEST_TYPE.FOUR_INPUTS, RANGE, SIZE, false, 1L},
				{TEST_TYPE.FOUR_INPUTS, RANGE, SIZE, true, 1L},
				{TEST_TYPE.FOUR_INPUTS, RANGE, SIZE2, true, 1L},
				
				// 3 inputs
				{TEST_TYPE.THREE_INPUTS1, RANGE, SIZE, false, -1},	// _seed is not used
				{TEST_TYPE.THREE_INPUTS1, RANGE, SIZE, true, -1},
				{TEST_TYPE.THREE_INPUTS1, RANGE, SIZE2, true, -1},
				
				{TEST_TYPE.THREE_INPUTS2, RANGE, SIZE, false, 1L}, // _replace is not used
				
				// 2 inputs
				{TEST_TYPE.TWO_INPUTS, RANGE, SIZE, false, -1},	 // _replace and _seed are not used
				{TEST_TYPE.TWO_INPUTS, RANGE, SIZE2, true, -1},
				
				// Error
				{TEST_TYPE.ERROR, RANGE, SIZE2, false, -1}
		};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_DIR, TEST_NAME,new String[]{"A"})); 
	}
	
	@Test
	public void testSample() {
		RUNTIME_PLATFORM platformOld = rtplatform;
		
		try
		{
			rtplatform = RUNTIME_PLATFORM.HYBRID;
			runSampleTest();
			rtplatform = RUNTIME_PLATFORM.SPARK;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			runSampleTest();
			rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
			runSampleTest();
			DMLScript.USE_LOCAL_SPARK_CONFIG = false;
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	private void runSampleTest() {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
	 	String HOME = SCRIPT_DIR + TEST_DIR;
		boolean exceptionExpected = false;

		if (test_type == TEST_TYPE.ERROR)
			exceptionExpected = true;

		switch (test_type) {
		case TWO_INPUTS:
			if (_range < _size)
				exceptionExpected = true;
			fullDMLScriptName = HOME + TEST_NAME + "2" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), HOME + OUTPUT_DIR + "A" };
			break;

		case THREE_INPUTS1:
			fullDMLScriptName = HOME + TEST_NAME + "3" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), (_replace ? "TRUE" : "FALSE"),
					HOME + OUTPUT_DIR + "A" };
			break;
		case THREE_INPUTS2:
			if (_range < _size)
				exceptionExpected = true;
			fullDMLScriptName = HOME + TEST_NAME + "3" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), Long.toString(_seed),
					HOME + OUTPUT_DIR + "A" };
			break;

		case FOUR_INPUTS:
		case ERROR:
			fullDMLScriptName = HOME + TEST_NAME + "4" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), (_replace ? "TRUE" : "FALSE"),
					Long.toString(_seed), HOME + OUTPUT_DIR + "A" };
			break;
		}

		loadTestConfiguration(config);

		runTest(true, exceptionExpected,
				(exceptionExpected ? DMLException.class : null), -1);

	}
	
}
