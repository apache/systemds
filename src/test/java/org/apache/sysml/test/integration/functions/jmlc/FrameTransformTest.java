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

import org.junit.Assert;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.api.jmlc.ResultVariables;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class FrameTransformTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "transform";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameTransformTest.class.getSimpleName() + "/";
	
	private final static int rows = 700;
	private final static int cols = 3;
	
	private final static int nRuns = 10;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Y" }) ); 
	}
	
	/*
	@Test
	public void testJMLCTransformDense() throws IOException {
		runJMLCReuseTest(TEST_NAME1, false, false);
	}
	
	@Test
	public void testJMLCTransformSparse() throws IOException {
		runJMLCReuseTest(TEST_NAME1, true, false);
	}
	
	@Test
	public void testJMLCTransformDenseReuse() throws IOException {
		runJMLCReuseTest(TEST_NAME1, false, true);
	}
	
	@Test
	public void testJMLCTransformSparseReuse() throws IOException {
		runJMLCReuseTest(TEST_NAME1, true, true);
	}
	*/

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 */
	@SuppressWarnings("unused")
	private void runJMLCReuseTest( String testname, boolean sparse, boolean modelReuse ) 
		throws IOException
	{	
		String TEST_NAME = testname;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//generate inputs
		double[][] Xd = TestUtils.round(getRandomMatrix(rows, cols, 0.51, 7.49, sparse?sparsity2:sparsity1, 1234));
		String[][] Xs = createFrameData(Xd);
		
		//run DML via JMLC
		ArrayList<double[][]> Yset = execDMLScriptviaJMLC( TEST_NAME, Xs, modelReuse );
		
		//check non-empty y
		for( double[][] data : Yset )
			Assert.assertEquals("Wrong result: "+data[0][0]+".", new Double(7), new Double(data[0][0]));
	}

	/**
	 * 
	 * @param X
	 * @return
	 * @throws DMLException
	 * @throws IOException
	 */
	private ArrayList<double[][]> execDMLScriptviaJMLC( String testname, String[][] X, boolean modelReuse) 
		throws IOException
	{
		Timing time = new Timing(true);
		
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		//establish connection to SystemML
		Connection conn = new Connection();
				
		try
		{
			//prepare input arguments
			HashMap<String,String> args = new HashMap<String,String>();
			args.put("$TRANSFORM_PATH", SCRIPT_DIR + TEST_DIR + "/tfmtd");
			args.put("$TRANSFORM_SPEC", "{ \"ids\": true ,\"recode\": [ 1, 2, 3] }");
			
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + testname + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, args, new String[]{"X"}, new String[]{"Y"}, false);
			
			if( modelReuse )
				pstmt.setFrame("X", X);
			
			//execute script multiple times
			for( int i=0; i<nRuns; i++ )
			{
				//bind input parameters
				if( !modelReuse )
					pstmt.setFrame("X", X);
				
				//execute script
				ResultVariables rs = pstmt.executeScript();
				
				//get output parameter
				double[][] Y = rs.getMatrix("Y");
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
	 * @param data
	 * @return
	 */
	private String[][] createFrameData(double[][] data) {
		String[][] ret = new String[data.length][];
		for( int i=0; i<data.length; i++ ) {
			String[] row = new String[data[i].length]; 
			for( int j=0; j<data[i].length; j++ )
				row[j] = "V"+String.valueOf(data[i][j]);
			ret[i] = row;
		}
		
		return ret;
	}
}