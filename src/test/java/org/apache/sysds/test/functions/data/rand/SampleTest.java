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

package org.apache.sysds.test.functions.data.rand;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Tests if Rand produces the same output, for a given set of parameters, across different (CP vs. MR) runtime platforms.   
 * 
 */
@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class SampleTest extends AutomatedTestBase 
{

	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_NAME = "Sample";
	private final static String TEST_CLASS_DIR = TEST_DIR + SampleTest.class.getSimpleName() + "/";
	
	private enum TEST_TYPE { FOUR_INPUTS, THREE_INPUTS1, THREE_INPUTS2, TWO_INPUTS, ERROR }
	
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
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"A"}));
	}
	
	@Test
	public void testSample() {
		ExecMode platformOld = rtplatform;
		
		try
		{
			rtplatform = ExecMode.HYBRID;
			runSampleTest();
			rtplatform = ExecMode.SPARK;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			runSampleTest();
			rtplatform = ExecMode.HYBRID;
			runSampleTest();
			DMLScript.USE_LOCAL_SPARK_CONFIG = false;
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	private void runSampleTest() {
		getAndLoadTestConfiguration(TEST_NAME);

	 	String HOME = SCRIPT_DIR + TEST_DIR;
		Class<?> expectedException = null;

		switch (test_type) {
		case TWO_INPUTS:
			if (_range < _size)
				expectedException = DMLRuntimeException.class;
			fullDMLScriptName = HOME + TEST_NAME + "2" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), output("A") };
			break;

		case THREE_INPUTS1:
			fullDMLScriptName = HOME + TEST_NAME + "3" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), (_replace ? "TRUE" : "FALSE"),
					output("A") };
			break;
		case THREE_INPUTS2:
			if (_range < _size)
				expectedException = LanguageException.class;
			fullDMLScriptName = HOME + TEST_NAME + "3" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), Long.toString(_seed),
					output("A") };
			break;

		case ERROR:
			expectedException = LanguageException.class;
		case FOUR_INPUTS:
			fullDMLScriptName = HOME + TEST_NAME + "4" + ".dml";
			programArgs = new String[] { "-args", Long.toString(_range),
					Long.toString(_size), (_replace ? "TRUE" : "FALSE"),
					Long.toString(_seed), output("A") };
			break;
		}

		runTest(expectedException);

	}
	
}
