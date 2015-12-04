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

package org.apache.sysml.test.integration.functions.io.binary;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class SerializeTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "SerializeTest";
	private final static String TEST_DIR = "functions/io/binary/";
	
	public static int rows1 = 746;
	public static int cols1 = 586;
	public static int cols2 = 4;
	
	private final static double eps = 1e-14;

	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "X" })   );  
	}
	
	@Test
	public void testEmptyBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.0 ); 
	}
	
	@Test
	public void testDenseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 1.0 ); 
	}
	
	@Test
	public void testDenseSparseBlock() 
	{ 
		runSerializeTest( rows1, cols2, 0.3 ); 
	}
	
	@Test
	public void testDenseUltraSparseBlock() 
	{ 
		runSerializeTest( rows1, cols2, 0.1 ); 
	}
	
	@Test
	public void testSparseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.1 ); 
	}
	
	@Test
	public void testSparseUltraSparseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.0001 ); 
	}

	private void runSerializeTest( int rows, int cols, double sparsity ) 
	{
		try
		{	
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "X",
											    HOME + OUTPUT_DIR + "X"    };
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, -1.0, 1.0, sparsity, 7); 
			MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, 1000, 1000);
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "X", OutputInfo.BinaryBlockOutputInfo, mc);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "X.mtd", ValueType.DOUBLE, mc, OutputInfo.BinaryBlockOutputInfo);
			
			runTest(true, false, null, -1); //mult 7
			
			//compare matrices 
			MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(HOME + OUTPUT_DIR + "X", InputInfo.BinaryBlockInputInfo, rows, cols, 1000, 1000);
			for( int i=0; i<mb.getNumRows(); i++ )
				for( int j=0; j<mb.getNumColumns(); j++ )
				{
					double val1 = mb.quickGetValue(i, j) * 7;
					double val2 = mb2.quickGetValue(i, j);
					Assert.assertEquals(val1, val2, eps);
				}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}