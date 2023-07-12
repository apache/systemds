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

package org.apache.sysds.test.functions.misc;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

/**
 *   
 */
public class ValueTypeCastingTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ValueTypeCastingTest.class.getSimpleName() + "/";

	private final static String TEST_NAME1 = "castDouble";
	private final static String TEST_NAME2 = "castInteger";
	private final static String TEST_NAME3 = "castBoolean";
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"R"}));
	}
	
	@Test
	public void testScalarDoubleToDouble() 
	{ 
		runTest( ValueType.FP64, ValueType.FP64, false, false ); 
	}
	
	@Test
	public void testScalarIntegerToDouble() 
	{ 
		runTest( ValueType.INT64, ValueType.FP64, false, false ); 
	}
	
	@Test
	public void testScalarBooleanToDouble() 
	{ 
		runTest( ValueType.BOOLEAN, ValueType.FP64, false, false );
	}
	
	@Test
	public void testMatrixDoubleToDouble() 
	{ 
		runTest( ValueType.FP64, ValueType.FP64, true, true ); 
	}
	
	@Test
	public void testScalarDoubleToInteger() 
	{ 
		runTest( ValueType.FP64, ValueType.INT64, false, false ); 
	}
	
	@Test
	public void testScalarIntegerToInteger() 
	{ 
		runTest( ValueType.INT64, ValueType.INT64, false, false ); 
	}
	
	@Test
	public void testScalarBooleanToInteger() 
	{ 
		runTest( ValueType.BOOLEAN, ValueType.INT64, false, false );
	}
	
	@Test
	public void testMatrixDoubleToInteger() 
	{ 
		runTest( ValueType.FP64, ValueType.INT64, true, true ); 
	}

	
	@Test
	public void testScalarDoubleToBoolean() 
	{ 
		runTest( ValueType.FP64, ValueType.BOOLEAN, false, false );
	}
	
	@Test
	public void testScalarIntegerToBoolean() 
	{ 
		runTest( ValueType.INT64, ValueType.BOOLEAN, false, false );
	}
	
	@Test
	public void testScalarBooleanToBoolean() 
	{ 
		runTest( ValueType.BOOLEAN, ValueType.BOOLEAN, false, false );
	}
	
	@Test
	public void testMatrixDoubleToBoolean() 
	{ 
		runTest( ValueType.FP64, ValueType.BOOLEAN, true, true );
	}
	
	private void runTest( ValueType vtIn, ValueType vtOut, boolean matrixInput, boolean exceptionExpected )
	{
		String TEST_NAME = null;
		switch( vtOut )
		{
			case FP64:  TEST_NAME = TEST_NAME1; break;
			case INT64: 	  TEST_NAME = TEST_NAME2; break;
			case BOOLEAN: TEST_NAME = TEST_NAME3; break;
			default: //do nothing
		}
		
		int numVals = (exceptionExpected ? 7 : 1);
		
		try
		{		
			getAndLoadTestConfiguration(TEST_NAME);
		    
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("V"), output("R") };
			
			//write input
			double[][] V = getRandomMatrix(numVals, numVals, 0, 1, 1.0, 7);
			double inVal = -1;
			if( matrixInput ){
				writeInputMatrix("V", V, false);	
				MatrixCharacteristics mc = new MatrixCharacteristics(numVals,numVals,1000,1000);
				HDFSTool.writeMetaDataFile(input("V.mtd"), vtIn, mc, FileFormat.TEXT);
			}
			else{
				HDFSTool.deleteFileIfExistOnHDFS(input("V"));
				switch( vtIn ) 
				{
					case FP64: 
						HDFSTool.writeDoubleToHDFS(V[0][0], input("V")); 
						inVal = V[0][0]; break;
					case INT64:    
						HDFSTool.writeIntToHDFS((int)V[0][0], input("V")); 
						inVal = ((int)V[0][0]); break;
					case BOOLEAN:
						HDFSTool.writeBooleanToHDFS(V[0][0]!=0, input("V")); 
						inVal = (V[0][0]!=0)?1:0; break;
					default: 
						//do nothing	
				}				
				HDFSTool.writeScalarMetaDataFile(input("V.mtd"), vtIn);
			}
			runTest(true, exceptionExpected, LanguageException.class, -1);
	        if( !exceptionExpected ){		        
		        //compare results
	        	String outName = output("R");
		        switch( vtOut ) {
					case FP64:  Assert.assertEquals(inVal, HDFSTool.readDoubleFromHDFSFile(outName), 1e-16); break;
					case INT64:     Assert.assertEquals((int) inVal, HDFSTool.readIntegerFromHDFSFile(outName)); break;
					case BOOLEAN: Assert.assertEquals(inVal!=0, HDFSTool.readBooleanFromHDFSFile(outName)); break;
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
