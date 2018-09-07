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

package org.apache.sysml.test.integration.functions.frame;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class FrameScalarCastingIntegratedTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FrameScalarCastingIntegratedTest.class.getSimpleName() + "/";
	private final static String TEST_NAME = "FrameScalarCast";
		
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}
	
	@Test
	public void testFrameStringCP0() { 
		runFrameScalarCastingTest(ValueType.STRING, RUNTIME_PLATFORM.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameLongCP0() { 
		runFrameScalarCastingTest(ValueType.INT, RUNTIME_PLATFORM.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameBooleanCP0() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, RUNTIME_PLATFORM.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameDoubleCP0() { 
		runFrameScalarCastingTest(ValueType.DOUBLE, RUNTIME_PLATFORM.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameStringCP1() { 
		runFrameScalarCastingTest(ValueType.STRING, RUNTIME_PLATFORM.HYBRID); 
	}
	
	@Test
	public void testFrameLongCP1() { 
		runFrameScalarCastingTest(ValueType.INT, RUNTIME_PLATFORM.HYBRID); 
	}
	
	@Test
	public void testFrameBooleanCP1() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, RUNTIME_PLATFORM.HYBRID); 
	}
	
	@Test
	public void testFrameDoubleCP1() { 
		runFrameScalarCastingTest(ValueType.DOUBLE, RUNTIME_PLATFORM.HYBRID); 
	}
	
	@Test
	public void testFrameStringCP2() { 
		runFrameScalarCastingTest(ValueType.STRING, RUNTIME_PLATFORM.HYBRID_SPARK); 
	}
	
	@Test
	public void testFrameLongCP2() { 
		runFrameScalarCastingTest(ValueType.INT, RUNTIME_PLATFORM.HYBRID_SPARK); 
	}
	
	@Test
	public void testFrameBooleanCP2() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, RUNTIME_PLATFORM.HYBRID_SPARK); 
	}
	
	@Test
	public void testFrameDoubleCP2() { 
		runFrameScalarCastingTest(ValueType.DOUBLE, RUNTIME_PLATFORM.HYBRID_SPARK); 
	}
	
	@Test
	public void testFrameStringSP() { 
		runFrameScalarCastingTest(ValueType.STRING, RUNTIME_PLATFORM.SPARK); 
	}
	
	@Test
	public void testFrameLongSP() { 
		runFrameScalarCastingTest(ValueType.INT, RUNTIME_PLATFORM.SPARK); 
	}
	
	@Test
	public void testFrameBooleanSP() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, RUNTIME_PLATFORM.SPARK); 
	}
	
	@Test
	public void testFrameDoubleSP() { 
		runFrameScalarCastingTest(ValueType.DOUBLE, RUNTIME_PLATFORM.SPARK); 
	}
	
	private void runFrameScalarCastingTest(ValueType vtIn, RUNTIME_PLATFORM et) 
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = et;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{		
			getAndLoadTestConfiguration(TEST_NAME);
		    
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", input("V"), output("R") };
			
			//generate input data
			switch( vtIn ) {
				case STRING: MapReduceTool.writeStringToHDFS("foo", input("V")); break;
				case INT: MapReduceTool.writeIntToHDFS(7, input("V")); break;
				case BOOLEAN: MapReduceTool.writeBooleanToHDFS(true, input("V")); break;
				case DOUBLE: MapReduceTool.writeDoubleToHDFS(7.3, input("V")); break;
				default: throw new RuntimeException("Unsupported type: "+vtIn);
			}
			MapReduceTool.writeScalarMetaDataFile(input("V")+".mtd", vtIn);
			
			//run tests
			runTest(true, false, null, -1);

			//compare output 
			Assert.assertEquals(readDMLMatrixFromHDFS("R").get(new CellIndex(1,1)), Double.valueOf(1));
			if( et != RUNTIME_PLATFORM.SPARK ) {
				Assert.assertTrue(Statistics.getNoOfCompiledSPInst()==0);
				Assert.assertTrue(Statistics.getNoOfExecutedSPInst()==0);
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
