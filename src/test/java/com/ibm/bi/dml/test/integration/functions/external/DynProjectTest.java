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

package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class DynProjectTest extends AutomatedTestBase 
{

	
	private final static String TEST_NAME = "DynProject";
	private final static String TEST_DIR = "functions/external/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1154;
	private final static int size = 104;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.3;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); 
	}

	
	@Test
	public void testProjectMatrixDense() 
	{
		runDynProjectTest(false, false);
	}
	
	@Test
	public void testProjectMatrixSparse() 
	{
		runDynProjectTest(false, true);
	}
	
	@Test
	public void testProjectVectorDense() 
	{
		runDynProjectTest(true, false);
	}
	
	@Test
	public void testProjectVectorSparse() 
	{
		runDynProjectTest(true, true);
	}

		
	/**
	 * 
	 * @param vector
	 * @param sparse
	 */
	private void runDynProjectTest( boolean vector, boolean sparse )
	{		
		double sparsity = sparse ? sparsity2 : sparsity1;
		int cols = vector ? 1 : rows;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "X" ,
				                        HOME + INPUT_DIR + "c",
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        Integer.toString(size),
				                        HOME + OUTPUT_DIR + "Y" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);
		
		long seed = System.nanoTime();
        double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
        double[][] c = round(getRandomMatrix(1, size, 1-0.49, rows+0.49, 1, seed));
        
        writeInputMatrix("X", X, true);
        writeInputMatrix("c", c, true);
		
		runTest(true, false, null, -1);
		runRScript(true);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("Y");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Y.mtx");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	private double[][] round(double[][] data) 
	{
		for(int i=0; i<data.length; i++)
			for(int j=0; j<data[i].length; j++)
				data[i][j]=Math.round(data[i][j]);
		return data;
	}
}