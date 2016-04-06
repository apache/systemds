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
import java.util.HashMap;
import java.util.Iterator;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.api.jmlc.ResultVariables;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

/**
 * 
 * 
 */
public class FrameReadMetaTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "transform3";
	private final static String TEST_DIR = "functions/jmlc/";
	
	private final static int rows = 300;
	private final static int cols = 9;	
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "F2" }) ); 
	}
	
	@Test
	public void testJMLCTransformDense() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, false);
	}
	
	@Test
	public void testJMLCTransformDenseReuse() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, true);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 */
	private void runJMLCReadMetaTest( String testname, boolean modelReuse ) 
		throws IOException
	{	
		String TEST_NAME = testname;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//establish connection to SystemML
		Connection conn = new Connection();
		
		//read meta data frame
		String spec = MapReduceTool.readStringFromHDFSFile(SCRIPT_DIR + TEST_DIR+"tfmtd_example/spec.json");
		FrameBlock M = conn.readTransformMetaDataFromFile(spec, SCRIPT_DIR + TEST_DIR+"tfmtd_example/");
		
		//generate data based on recode maps
		HashMap<String,Long>[] RC = getRecodeMaps(M);
		double[][] X = generateData(rows, cols, RC);
		String[][] F = null;
		
		try
		{
			//prepare input arguments
			HashMap<String,String> args = new HashMap<String,String>();
			args.put("$TRANSFORM_SPEC", spec);
			
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + testname + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, args, new String[]{"X","M"}, new String[]{"F"}, false);
			
			if( modelReuse )
				pstmt.setFrame("M", M, true);
			
			//execute script multiple times (2 runs)
			for( int i=0; i<2; i++ )
			{
				//bind input parameters
				if( !modelReuse )
					pstmt.setFrame("M", M, false);
				pstmt.setMatrix("X", X);
				
				//execute script
				ResultVariables rs = pstmt.executeScript();
				
				//get output parameter
				F = rs.getFrame("F");
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new IOException(ex);
		}
		finally
		{
			if( conn != null )
				conn.close();
		}
		
		//check correct result 
		//for all generated data, probe recode maps and compare versus output
		for( int i=0; i<rows; i++ ) 
			for( int j=0; j<cols; j++ ) 
				if( RC[j] != null ) {
					Assert.assertEquals("Wrong result: "+F[i][j]+".", 
							Double.valueOf(X[i][j]), 
							Double.valueOf(RC[j].get(F[i][j]).toString()));
				}	
	}

	/**
	 * 
	 * @param M
	 * @return
	 */
	@SuppressWarnings("unchecked")
	private HashMap<String,Long>[] getRecodeMaps(FrameBlock M) {
		HashMap<String,Long>[] ret = new HashMap[M.getNumColumns()];
		Iterator<Object[]> iter = M.getObjectRowIterator();
		while( iter.hasNext() ) {
			Object[] tmp = iter.next();
			for( int j=0; j<tmp.length; j++ ) 
				if( tmp[j] != null ) {
					if( ret[j] == null )
						ret[j] = new HashMap<String,Long>();
					String[] parts = tmp[j].toString().split(Lop.DATATYPE_PREFIX);
					ret[j].put(parts[0], Long.parseLong(parts[1]));
				}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param rows
	 * @param cols
	 * @param RC
	 * @return
	 */
	private double[][] generateData(int rows, int cols, HashMap<String,Long>[] RC) {
		double[][] ret = new double[rows][cols];
		for( int i=0; i<rows; i++ ) 
			for( int j=0; j<cols; j++ ) 
				if( RC[j] != null ) {
					ret[i][j] = RC[j].values().toArray(new Long[0])[i%RC[j].size()];
				}
		
		return ret;
	}
}