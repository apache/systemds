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

import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.api.jmlc.ResultVariables;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class MulticlassSVMScoreTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "m-svm-score";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String MODEL_FILE = "sentiment_model.mtx";
	private final static double eps = 1e-10;
	private final static String TEST_CLASS_DIR = TEST_DIR + MulticlassSVMScoreTest.class.getSimpleName() + "/";
	
	private final static int rows = 107;
	private final static int cols = 46; //fixed
	
	private final static int nRuns = 3;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "predicted_y" }) ); 
	}
	
	@Test
	public void testJMLCMulticlassScoreDense() throws IOException {
		runJMLCMulticlassTest(false, false);
	}
	
	@Test
	public void testJMLCMulticlassScoreSparse() throws IOException {
		runJMLCMulticlassTest(true, false);
	}
	
	@Test
	public void testJMLCMulticlassScoreDenseFlags() throws IOException {
		runJMLCMulticlassTest(false, true);
	}
	
	@Test
	public void testJMLCMulticlassScoreSparseFlags() throws IOException {
		runJMLCMulticlassTest(true, true);
	}

	private void runJMLCMulticlassTest( boolean sparse, boolean flags ) 
		throws IOException
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//generate inputs
		ArrayList<double[][]> Xset = generateInputs(nRuns, rows, cols, sparse?sparsity2:sparsity1, 7);
		
		//run DML via JMLC
		ArrayList<double[][]> Yset = execDMLScriptviaJMLC( Xset, flags );
		
		//write out R input and model once
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(SCRIPT_DIR + TEST_DIR + MODEL_FILE,
			InputInfo.TextCellInputInfo, rows, cols, 1000, 1000);
		writeInputMatrix("X", Xset.get(0), true);
		writeInputMatrix("W", DataConverter.convertToDoubleMatrix(mb), true);
		
		//run R test once
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = getRCmd(inputDir(), expectedDir());
		runRScript(true);
		
		//read and convert R output
		HashMap<CellIndex, Double> rfile = readRMatrixFromFS("predicted_y");
		double[][] expected = TestUtils.convertHashMapToDoubleArray(rfile, rows, 1);
		
		//for each input data set compare results
		for( int i=0; i<nRuns; i++ )
			TestUtils.compareMatrices(expected, Yset.get(i), rows, 1, eps);
	}

	private static ArrayList<double[][]> execDMLScriptviaJMLC(ArrayList<double[][]> X, boolean flags) 
		throws IOException
	{
		Timing time = new Timing(true);
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		try( Connection conn = !flags ? new Connection():
			new Connection(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS,
			ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR,
			ConfigType.ALLOW_DYN_RECOMPILATION) )
		{
			// For now, JMLC pipeline only allows dml
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
		catch(Exception ex) {
			throw new IOException(ex);
		}
		
		System.out.println("JMLC scoring w/ "+nRuns+" runs in "+time.stop()+"ms.");
		
		return ret;
	}

	private ArrayList<double[][]> generateInputs( int num, int rows, int cols, double sparsity, int seed ) {
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		for( int i=0; i<num; i++ )
			ret.add(getRandomMatrix(rows, cols, -1, 1, sparsity, seed));
		return ret;
	}
}
