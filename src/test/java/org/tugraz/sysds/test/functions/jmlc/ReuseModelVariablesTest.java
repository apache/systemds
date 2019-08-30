/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.test.functions.jmlc;

import java.io.IOException;
import java.util.ArrayList;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.jmlc.Connection;
import org.tugraz.sysds.api.jmlc.PreparedScript;
import org.tugraz.sysds.api.jmlc.ResultVariables;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

public class ReuseModelVariablesTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "reuse-glm-predict";
	private final static String TEST_NAME2 = "reuse-msvm-predict";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String MODEL_FILE = "sentiment_model.mtx";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReuseModelVariablesTest.class.getSimpleName() + "/";
	
	private final static int rows = 107;
	private final static int cols = 46; //fixed
	
	private final static int nRuns = 10;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "predicted_y" }) ); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "predicted_y" }) );
	}
	
	@Test
	public void testJMLCScoreGLMDense() throws IOException {
		runJMLCReuseTest(TEST_NAME1, false, false);
	}
	
	@Test
	public void testJMLCScoreGLMSparse() throws IOException {
		runJMLCReuseTest(TEST_NAME1, true, false);
	}
	
	@Test
	public void testJMLCScoreGLMDenseReuse() throws IOException {
		runJMLCReuseTest(TEST_NAME1, false, true);
	}
	
	@Test
	public void testJMLCScoreGLMSparseReuse() throws IOException {
		runJMLCReuseTest(TEST_NAME1, true, true);
	}
	
	@Test
	public void testJMLCScoreMSVMDense() throws IOException {
		runJMLCReuseTest(TEST_NAME2, false, false);
	}
	
	@Test
	public void testJMLCScoreMSVMSparse() throws IOException {
		runJMLCReuseTest(TEST_NAME2, true, false);
	}
	
	@Test
	public void testJMLCScoreMSVMDenseReuse() throws IOException {
		runJMLCReuseTest(TEST_NAME2, false, true);
	}
	
	@Test
	public void testJMLCScoreMSVMSparseReuse() throws IOException {
		runJMLCReuseTest(TEST_NAME2, true, true);
	}

	private void runJMLCReuseTest( String testname, boolean sparse, boolean modelReuse ) 
		throws IOException
	{	
		String TEST_NAME = testname;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//generate inputs
		ArrayList<double[][]> Xset = generateInputs(nRuns, rows, cols, sparse?sparsity2:sparsity1); 
		
		//run DML via JMLC
		ArrayList<double[][]> Yset = execDMLScriptviaJMLC( TEST_NAME, Xset, modelReuse );
		
		//check non-empty y
		Assert.assertEquals(Xset.size(), Yset.size());
	}

	private static ArrayList<double[][]> execDMLScriptviaJMLC( String testname, ArrayList<double[][]> X, boolean modelReuse) 
		throws IOException
	{
		Timing time = new Timing(true);
		
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		//establish connection to SystemDS
		Connection conn = new Connection();
		
		try
		{
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + testname + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, new String[]{"X","W"}, new String[]{"predicted_y"});
			
			//read model
			String modelData = conn.readScript(SCRIPT_DIR + TEST_DIR + MODEL_FILE );
			double[][] W = conn.convertToDoubleMatrix(modelData, rows, cols); 
			
			if( modelReuse )
				pstmt.setMatrix("W", W, true);
			
			//execute script multiple times
			for( int i=0; i<nRuns; i++ )
			{
				//bind input parameters
				if( !modelReuse )
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
		finally {
			IOUtilFunctions.closeSilently(conn);
		}
		
		System.out.println("JMLC scoring w/ "+nRuns+" runs in "+time.stop()+"ms.");
		
		return ret;
	}

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