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
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.jmlc.Connection;
import org.tugraz.sysds.api.jmlc.PreparedScript;
import org.tugraz.sysds.api.jmlc.ResultVariables;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class FrameLeftIndexingTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "transform5";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameLeftIndexingTest.class.getSimpleName() + "/";
	
	private final static int rows = 700;
	private final static int cols = 3;
	
	private final static int nRuns = 2;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "F2" }) ); 
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
		double[][] Fd = TestUtils.round(getRandomMatrix(rows, cols, 0.51, 7.49, sparse?sparsity2:sparsity1, 1234));
		String[][] F1s = FrameTransformTest.createFrameData(Fd);
		String[][] Ms = FrameTransformTest.createRecodeMaps(F1s);
		
		//run DML via JMLC
		ArrayList<String[][]> F2set = execDMLScriptviaJMLC( TEST_NAME, F1s, Ms, modelReuse );
		
		//check correct result 
		for( String[][] data : F2set )
			for( int i=0; i<F1s.length; i++ )
				for( int j=0; j<F1s[i].length; j++ )
					Assert.assertEquals("Wrong result: "+data[i][j]+".", data[i][j], F1s[i][j]);
	}

	private static ArrayList<String[][]> execDMLScriptviaJMLC( String testname, String[][] F1, String[][] M, boolean modelReuse) 
		throws IOException
	{
		Timing time = new Timing(true);
		
		ArrayList<String[][]> ret = new ArrayList<String[][]>();
		
		//establish connection to SystemDS
		Connection conn = new Connection();
		
		try
		{
			//prepare input arguments
			HashMap<String,String> args = new HashMap<String,String>();
			args.put("$TRANSFORM_SPEC1", "{ \"ids\": true ,\"recode\": [ 1, 2] }");
			args.put("$TRANSFORM_SPEC2", "{ \"ids\": true ,\"recode\": [ 1] }");
			
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + testname + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, args, new String[]{"F1","M"}, new String[]{"F2"});
			
			if( modelReuse )
				pstmt.setFrame("M", M, true);
			
			//execute script multiple times
			for( int i=0; i<nRuns; i++ )
			{
				//bind input parameters
				if( !modelReuse )
					pstmt.setFrame("M", M);
				pstmt.setFrame("F1", F1);
				
				//execute script
				ResultVariables rs = pstmt.executeScript();
				
				//get output parameter
				String[][] Y = rs.getFrame("F2");
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
}