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

package org.apache.sysml.test.integration.functions.misc;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class ValueTypeMatrixScalarBuiltinTest extends AutomatedTestBase
{	
	private final static String TEST_NAME1 = "ValueTypeMaxLeftScalar";
	private final static String TEST_NAME2 = "ValueTypeMaxRightScalar";
	private final static String TEST_NAME3 = "ValueTypeLogLeftScalar";
	private final static String TEST_NAME4 = "ValueTypeLogRightScalar";
	private final static String TEST_NAME5 = "ValueTypePredLeftScalar";
	private final static String TEST_NAME6 = "ValueTypePredRightScalar";
	
	private final static String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ValueTypeMatrixScalarBuiltinTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {}));
	}

	@Test
	public void testValueTypeMaxLeftScalarDouble() { 
		runTest(TEST_NAME1, ValueType.DOUBLE); 
	}
	
	@Test
	public void testValueTypeMaxLeftScalarInt() { 
		runTest(TEST_NAME1, ValueType.INT); 
	}
	
	@Test
	public void testValueTypeMaxRightScalarDouble() { 
		runTest(TEST_NAME2, ValueType.DOUBLE); 
	}
	
	@Test
	public void testValueTypeMaxRightScalarInt() { 
		runTest(TEST_NAME2, ValueType.INT); 
	}
	
	@Test
	public void testValueTypeLogLeftScalarDouble() { 
		runTest(TEST_NAME3, ValueType.DOUBLE); 
	}
	
	@Test
	public void testValueTypeLogLeftScalarInt() { 
		runTest(TEST_NAME3, ValueType.INT); 
	}
	
	@Test
	public void testValueTypeLogRightScalarDouble() { 
		runTest(TEST_NAME4, ValueType.DOUBLE); 
	}
	
	@Test
	public void testValueTypeLogRightScalarInt() { 
		runTest(TEST_NAME4, ValueType.INT); 
	}
	
	@Test
	public void testValueTypePredLeftScalarDouble() { 
		runTest(TEST_NAME5, ValueType.DOUBLE); 
	}
	
	@Test
	public void testValueTypePredLeftScalarInt() { 
		runTest(TEST_NAME5, ValueType.INT); 
	}
	
	@Test
	public void testValueTypePredRightScalarDouble() { 
		runTest(TEST_NAME6, ValueType.DOUBLE); 
	}
	
	@Test
	public void testValueTypePredRightScalarInt() { 
		runTest(TEST_NAME6, ValueType.INT); 
	}
	
	private void runTest(String testName, ValueType vtIn) 
	{
		loadTestConfiguration(getTestConfiguration(testName));
		
		//setup arguments and run test
		String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + testName + ".dml";
		programArgs = new String[]{"-args", 
			vtIn==ValueType.DOUBLE ? "7.7" : "7", output("R")};
		runTest(true, false, null, -1);
		
		//check output value type
		ValueType vtOut = readDMLMetaDataValueType("R");
		Assert.assertTrue("Wrong output value type: " + 
			vtOut.name(), vtOut.equals(ValueType.DOUBLE));
	}
}
