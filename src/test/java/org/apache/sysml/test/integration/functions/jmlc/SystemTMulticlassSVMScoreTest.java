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

package org.apache.sysml.test.integration.functions.jmlc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.api.jmlc.ResultVariables;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class SystemTMulticlassSVMScoreTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "m-svm-score";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String MODEL_FILE = "sentiment_model.mtx";
	private final static double eps = 1e-10;
	
	private final static int rows = 107;
	private final static int cols = 46; //fixed
	
	private final static int nRuns = 10;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "predicted_y" })   ); 
	}

	
	@Test
	public void testJMLCMulticlassScoreDense() 
		throws IOException
	{
		//should apply diag_mm rewrite
		runJMLCMulticlassTest(false);
	}
	
	@Test
	public void testJMLCMulticlassScoreSparse() 
		throws IOException
	{
		//should apply diag_mm rewrite
		runJMLCMulticlassTest(false);
	}
	

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 */
	private void runJMLCMulticlassTest( boolean sparse ) 
		throws IOException
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
	
		//generate inputs
		ArrayList<double[][]> Xset = generateInputs(nRuns, rows, cols, sparse?sparsity2:sparsity1); 
		
		//run DML via JMLC
		ArrayList<double[][]> Yset = execDMLScriptviaJMLC( Xset );
		
		//run R and compare results to DML result
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//write model data once
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(SCRIPT_DIR + TEST_DIR + MODEL_FILE, 
				                       InputInfo.TextCellInputInfo, rows, cols, 1000, 1000);
		double[][] W = DataConverter.convertToDoubleMatrix( mb );
		writeInputMatrix("W", W, true);
		
		//for each input data set
		for( int i=0; i<nRuns; i++ )
		{
			//write input data
			writeInputMatrix("X", Xset.get(i), true);	
			
			//run the R script
			runRScript(true); 
			
			//compare results
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("predicted_y");
			double[][] expected = TestUtils.convertHashMapToDoubleArray(rfile, rows, 1);
			
			TestUtils.compareMatrices(expected, Yset.get(i), rows, 1, eps);	
		}
	}

	/**
	 * 
	 * @param X
	 * @return
	 * @throws DMLException
	 * @throws IOException
	 */
	private ArrayList<double[][]> execDMLScriptviaJMLC( ArrayList<double[][]> X) 
		throws IOException
	{
		Timing time = new Timing(true);
		
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		//establish connection to SystemML
		Connection conn = new Connection();
				
		try
		{
			// Note for Matthias: For now, JMLC pipeline only allows dml
			boolean parsePyDML = false;
			
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, new String[]{"X","W"}, new String[]{"predicted_y"}, parsePyDML);
			
			//read model
			String modelData = conn.readScript(SCRIPT_DIR + TEST_DIR + MODEL_FILE );
			double[][] W = conn.convertToDoubleMatrix(modelData, rows, cols); 
			
			//execute script multiple times
			for( int i=0; i<nRuns; i++ )
			{
				//bind input parameters
				pstmt.setMatrix("W", W);
				pstmt.setMatrix("X", X.get(i));
				
				//execute script
				ResultVariables rs = pstmt.executeScript();
				
				//get output parameter
				double[][] Y = rs.getMatrix("predicted_y");
				ret.add(Y); //keep result for comparison
			}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new IOException(ex);
		}
		finally
		{
			if( conn != null )
				conn.close();
		}
		
		System.out.println("JMLC scoring w/ "+nRuns+" runs in "+time.stop()+"ms.");
		
		return ret;
	}
	
	/**
	 * 
	 * @param num
	 * @param rows
	 * @param cols
	 * @param sparsity
	 * @return
	 */
	private ArrayList<double[][]> generateInputs( int num, int rows, int cols, double sparsity )
	{
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		for( int i=0; i<num; i++ )
		{
			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, System.nanoTime());
			ret.add(X);
		}
		
		return ret;
	}
	
}