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

package org.apache.sysds.test.functions.frame;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

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
		runFrameScalarCastingTest(ValueType.STRING, ExecMode.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameLongCP0() { 
		runFrameScalarCastingTest(ValueType.INT64, ExecMode.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameBooleanCP0() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testFrameDoubleCP0() { 
		runFrameScalarCastingTest(ValueType.FP64, ExecMode.SINGLE_NODE); 
	}
	
	@Test
	public void testFrameStringCP1() { 
		runFrameScalarCastingTest(ValueType.STRING, ExecMode.HYBRID); 
	}
	
	@Test
	public void testFrameLongCP1() { 
		runFrameScalarCastingTest(ValueType.INT64, ExecMode.HYBRID); 
	}
	
	@Test
	public void testFrameBooleanCP1() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, ExecMode.HYBRID);
	}
	
	@Test
	public void testFrameDoubleCP1() { 
		runFrameScalarCastingTest(ValueType.FP64, ExecMode.HYBRID); 
	}
	
	@Test
	public void testFrameStringCP2() { 
		runFrameScalarCastingTest(ValueType.STRING, ExecMode.HYBRID); 
	}
	
	@Test
	public void testFrameLongCP2() { 
		runFrameScalarCastingTest(ValueType.INT64, ExecMode.HYBRID); 
	}
	
	@Test
	public void testFrameBooleanCP2() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, ExecMode.HYBRID);
	}
	
	@Test
	public void testFrameDoubleCP2() { 
		runFrameScalarCastingTest(ValueType.FP64, ExecMode.HYBRID); 
	}
	
	@Test
	public void testFrameStringSP() { 
		runFrameScalarCastingTest(ValueType.STRING, ExecMode.SPARK); 
	}
	
	@Test
	public void testFrameLongSP() { 
		runFrameScalarCastingTest(ValueType.INT64, ExecMode.SPARK); 
	}
	
	@Test
	public void testFrameBooleanSP() { 
		runFrameScalarCastingTest(ValueType.BOOLEAN, ExecMode.SPARK);
	}
	
	@Test
	public void testFrameDoubleSP() { 
		runFrameScalarCastingTest(ValueType.FP64, ExecMode.SPARK); 
	}
	
	private void runFrameScalarCastingTest(ValueType vtIn, ExecMode et) 
	{
		ExecMode platformOld = rtplatform;
		rtplatform = et;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		setOutputBuffering(true);
		try
		{		
			getAndLoadTestConfiguration(TEST_NAME);
		    
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-explain", "-args", input("V"), output("R") };
			
			//generate input data
			switch( vtIn ) {
				case STRING: HDFSTool.writeStringToHDFS("foo", input("V")); break;
				case INT64: HDFSTool.writeIntToHDFS(7, input("V")); break;
				case BOOLEAN: HDFSTool.writeBooleanToHDFS(true, input("V")); break;
				case FP64: HDFSTool.writeDoubleToHDFS(7.3, input("V")); break;
				default: throw new RuntimeException("Unsupported type: "+vtIn);
			}
			HDFSTool.writeScalarMetaDataFile(input("V")+".mtd", vtIn);
			
			//run tests
			runTest(null);

			//compare output 
			Assert.assertEquals(readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1)), Double.valueOf(1));
			if( et != ExecMode.SPARK ) {
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
