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

package org.apache.sysml.test.integration.functions.caching;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

public class CachingPWriteExportTest extends AutomatedTestBase 
{

	
	private final static String TEST_NAME = "export";
	private final static String TEST_DIR = "functions/caching/";

	private final static int rows = (int)Hop.CPThreshold-1;
	private final static int cols = (int)Hop.CPThreshold-1;    
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "V" })   ); 
	}
	
	@Test
	public void testExportReadWrite() 
	{
		runTestExport( "binary" );
	}
	
	@Test
	public void testExportCopy() 
	{
		runTestExport( "text" );
	}

	
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runTestExport( String outputFormat )
	{				
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "V",
				                        outputFormat };
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
		writeInputMatrix("V", V, true); //always text
		writeExpectedMatrix("V", V);
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		
		double[][] Vp = null;
		try
		{
			InputInfo ii = null;
			if( outputFormat.equals("binary") )
				ii = InputInfo.BinaryBlockInputInfo;
			else
				ii = InputInfo.TextCellInputInfo;
			
			MatrixBlock mb = DataConverter.readMatrixFromHDFS(HOME + OUTPUT_DIR + "V", ii, rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, sparsity);
			Vp = DataConverter.convertToDoubleMatrix(mb);
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);		}
		
		//compare
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				if( V[i][j]!=Vp[i][j] )
					//System.out.println("Wrong value i="+i+", j="+j+", value1="+V[i][j]+", value2="+Vp[i][j]);
					Assert.fail("Wrong value i="+i+", j="+j+", value1="+V[i][j]+", value2="+Vp[i][j]);
	}
}