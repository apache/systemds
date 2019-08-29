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

package org.tugraz.sysds.test.functions.jmlc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.jmlc.Connection;
import org.tugraz.sysds.api.jmlc.PreparedScript;
import org.tugraz.sysds.api.jmlc.ResultVariables;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class FrameTransformTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "transform";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameTransformTest.class.getSimpleName() + "/";
	
	private final static int rows = 700;
	private final static int cols = 3;
	
	private final static int nRuns = 2;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Y" }) ); 
	}
	
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

	private void runJMLCReuseTest( String testname, boolean sparse, boolean modelReuse ) 
		throws IOException
	{	
		String TEST_NAME = testname;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//generate inputs
		double[][] Xd = TestUtils.round(getRandomMatrix(rows, cols, 0.51, 7.49, sparse?sparsity2:sparsity1, 1234));
		setColumnValue(Xd, 2, 3); //create ragged meta frame
		String[][] Xs = createFrameData(Xd);
		String[][] Ms = createRecodeMaps(Xs);
		
		//run DML via JMLC
		ArrayList<double[][]> Yset = execDMLScriptviaJMLC( TEST_NAME, Xs, Ms, modelReuse );
		
		//check correct result (nnz 7 + 0 -> 8 distinct vals)
		for( double[][] data : Yset )
			Assert.assertEquals("Wrong result: "+data[0][0]+".", new Double(8), new Double(data[0][0]));
	}

	private static ArrayList<double[][]> execDMLScriptviaJMLC( String testname, String[][] X, String[][] M, boolean modelReuse) 
		throws IOException
	{
		Timing time = new Timing(true);
		
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		//establish connection to SystemDS
		Connection conn = new Connection();
		
		try
		{
			//prepare input arguments
			HashMap<String,String> args = new HashMap<String,String>();
			args.put("$TRANSFORM_SPEC", "{ \"ids\": true ,\"recode\": [ 1, 2, 3] }");
			
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + testname + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, args, new String[]{"X","M"}, new String[]{"Y"});
			
			if( modelReuse )
				pstmt.setFrame("M", M, true);
			
			//execute script multiple times
			for( int i=0; i<nRuns; i++ )
			{
				//bind input parameters
				if( !modelReuse )
					pstmt.setFrame("M", M);
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
		finally {
			IOUtilFunctions.closeSilently(conn);
		}
		
		System.out.println("JMLC scoring w/ "+nRuns+" runs in "+time.stop()+"ms.");
		
		return ret;
	}
	
	protected static String[][] createFrameData(double[][] data) {
		return createFrameData(data, "V");
	}

	protected static String[][] createFrameData(double[][] data, String prefix) {
		String[][] ret = new String[data.length][];
		for( int i=0; i<data.length; i++ ) {
			String[] row = new String[data[i].length]; 
			for( int j=0; j<data[i].length; j++ )
				row[j] = prefix+String.valueOf(data[i][j]);
			ret[i] = row;
		}
		
		return ret;
	}
	
	protected static String[][] createRecodeMaps(String[][] data) {
		//create maps per column
		ArrayList<HashMap<String,Integer>> map = new ArrayList<HashMap<String,Integer>>(); 
		for( int j=0; j<data[0].length; j++ )
			map.add(new HashMap<String,Integer>());
		//create recode maps per column
		for( int i=0; i<data.length; i++ ) {
			for( int j=0; j<data[i].length; j++ )
				if( !map.get(j).containsKey(data[i][j]) )
					map.get(j).put(data[i][j], map.get(j).size()+1);
		}
		//determine max recode map size
		int max = 0;
		for( int j=0; j<data[0].length; j++ )
			max = Math.max(max, map.get(j).size());
		
		//allocate output
		String[][] ret = new String[max][];
		for( int i=0; i<max; i++ )
			ret[i] = new String[data[0].length];
		
		//create frame of recode maps
		for( int j=0; j<data[0].length; j++) {
			int i = 0;
			for( Entry<String, Integer> e : map.get(j).entrySet() )
				ret[i++][j] = e.getKey()+Lop.DATATYPE_PREFIX+e.getValue();
		}
		
		return ret;
	}
	
	protected void setColumnValue(double[][] data, int col, double val) {
		for( int i=0; i<data.length; i++ )
			data[i][col] = val;
	} 
}