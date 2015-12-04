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

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

/**
 *   
 */
public class ValueTypeCastingTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "castDouble";
	private final static String TEST_NAME2 = "castInteger";
	private final static String TEST_NAME3 = "castBoolean";
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {"R"}));
	}
	
	@Test
	public void testScalarDoubleToDouble() 
	{ 
		runTest( ValueType.DOUBLE, ValueType.DOUBLE, false, false ); 
	}
	
	@Test
	public void testScalarIntegerToDouble() 
	{ 
		runTest( ValueType.INT, ValueType.DOUBLE, false, false ); 
	}
	
	@Test
	public void testScalarBooleanToDouble() 
	{ 
		runTest( ValueType.BOOLEAN, ValueType.DOUBLE, false, false ); 
	}
	
	@Test
	public void testMatrixDoubleToDouble() 
	{ 
		runTest( ValueType.DOUBLE, ValueType.DOUBLE, true, true ); 
	}
	
	@Test
	public void testScalarDoubleToInteger() 
	{ 
		runTest( ValueType.DOUBLE, ValueType.INT, false, false ); 
	}
	
	@Test
	public void testScalarIntegerToInteger() 
	{ 
		runTest( ValueType.INT, ValueType.INT, false, false ); 
	}
	
	@Test
	public void testScalarBooleanToInteger() 
	{ 
		runTest( ValueType.BOOLEAN, ValueType.INT, false, false ); 
	}
	
	@Test
	public void testMatrixDoubleToInteger() 
	{ 
		runTest( ValueType.DOUBLE, ValueType.INT, true, true ); 
	}

	
	@Test
	public void testScalarDoubleToBoolean() 
	{ 
		runTest( ValueType.DOUBLE, ValueType.BOOLEAN, false, false ); 
	}
	
	@Test
	public void testScalarIntegerToBoolean() 
	{ 
		runTest( ValueType.INT, ValueType.BOOLEAN, false, false ); 
	}
	
	@Test
	public void testScalarBooleanToBoolean() 
	{ 
		runTest( ValueType.BOOLEAN, ValueType.BOOLEAN, false, false ); 
	}
	
	@Test
	public void testMatrixDoubleToBoolean() 
	{ 
		runTest( ValueType.DOUBLE, ValueType.BOOLEAN, true, true ); 
	}
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runTest( ValueType vtIn, ValueType vtOut, boolean matrixInput, boolean exceptionExpected ) 
	{
		String TEST_NAME = null;
		switch( vtOut )
		{
			case DOUBLE:  TEST_NAME = TEST_NAME1; break;
			case INT: 	  TEST_NAME = TEST_NAME2; break;
			case BOOLEAN: TEST_NAME = TEST_NAME3; break;
			default: //do nothing
		}
		
		int numVals = (exceptionExpected ? 7 : 1);
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
								                HOME + OUTPUT_DIR + "R", };
			
			loadTestConfiguration(config);
			
			//write input
			double[][] V = getRandomMatrix(numVals, numVals, 0, 1, 1.0, 7);
			double inVal = -1;
			if( matrixInput ){
				writeInputMatrix("V", V, false);	
				MatrixCharacteristics mc = new MatrixCharacteristics(numVals,numVals,1000,1000);
				MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "V.mtd", vtIn, mc, OutputInfo.TextCellOutputInfo);
			}
			else{
				MapReduceTool.deleteFileIfExistOnHDFS(HOME + INPUT_DIR + "V");
				switch( vtIn ) 
				{
					case DOUBLE: 
						MapReduceTool.writeDoubleToHDFS(V[0][0], HOME + INPUT_DIR + "V"); 
						inVal = V[0][0]; break;
					case INT:    
						MapReduceTool.writeIntToHDFS((int)V[0][0], HOME + INPUT_DIR + "V"); 
						inVal = ((int)V[0][0]); break;
					case BOOLEAN: 
						MapReduceTool.writeBooleanToHDFS(V[0][0]!=0, HOME + INPUT_DIR + "V"); 
						inVal = (V[0][0]!=0)?1:0; break;
					default: 
						//do nothing	
				}				
				MapReduceTool.writeScalarMetaDataFile(HOME + INPUT_DIR + "V.mtd", vtIn);
			}
			
			
			//run tests
	        runTest(true, exceptionExpected, DMLException.class, -1);
	        
	        if( !exceptionExpected ){		        
		        //compare results
	        	String outName = HOME + OUTPUT_DIR + "R";
		        switch( vtOut ) {
					case DOUBLE:  Assert.assertEquals(inVal, MapReduceTool.readDoubleFromHDFSFile(outName), 1e-16); break;
					case INT:     Assert.assertEquals((int) inVal, MapReduceTool.readIntegerFromHDFSFile(outName)); break;
					case BOOLEAN: Assert.assertEquals(inVal!=0, MapReduceTool.readBooleanFromHDFSFile(outName)); break;
					default: //do nothing
		        }
	        }
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
