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
 
package org.apache.sysds.test.functions.jmlc;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.api.jmlc.ResultVariables;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

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
	public void testJMLCTransformDenseSpec() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, false, false, true);
	}
	
	@Test
	public void testJMLCTransformDenseReuseSpec() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, true, false, true);
	}
	
	@Test
	public void testJMLCTransformDense() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, false, false, false);
	}
	
	@Test
	public void testJMLCTransformDenseReuse() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, true, false, false);
	}
	
	@Test
	public void testJMLCTransformDenseReadFrame() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, false, true, false);
	}
	
	@Test
	public void testJMLCTransformDenseReuseReadFrame() throws IOException {
		runJMLCReadMetaTest(TEST_NAME1, true, true, false);
	}

	private void runJMLCReadMetaTest( String testname, boolean modelReuse, boolean readFrame, boolean useSpec ) 
		throws IOException
	{	
		String TEST_NAME = testname;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//establish connection to SystemDS
		Connection conn = new Connection();
		
		//read meta data frame 
		String spec = HDFSTool.readStringFromHDFSFile(SCRIPT_DIR + TEST_DIR+"tfmtd_example2/spec.json");
		FrameBlock M = readFrame ?
				DataConverter.convertToFrameBlock(conn.readStringFrame(SCRIPT_DIR + TEST_DIR+"tfmtd_frame_example/tfmtd_frame")) : 
				conn.readTransformMetaDataFromFile(spec, SCRIPT_DIR + TEST_DIR+"tfmtd_example2/");
		
		try
		{
			//generate data based on recode maps
			HashMap<String,Long>[] RC = getRecodeMaps(spec, M);
			double[][] X = generateData(rows, cols, RC);
			String[][] F = null;
			
			//prepare input arguments
			HashMap<String,String> args = new HashMap<>();
			args.put("$TRANSFORM_SPEC", spec);
			
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + testname + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, args, new String[]{"X","M"}, new String[]{"F"});
			
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
		catch(Exception ex) {
			ex.printStackTrace();
			throw new IOException(ex);
		}
		finally {
			IOUtilFunctions.closeSilently(conn);
		}
	}

	@SuppressWarnings("unchecked")
	private static HashMap<String,Long>[] getRecodeMaps(String spec, FrameBlock M) {
		List<Integer> collist = Arrays.asList(ArrayUtils.toObject(
			TfMetaUtils.parseJsonIDList(spec, M.getColumnNames(), TfMethod.RECODE.toString())));
		HashMap<String,Long>[] ret = new HashMap[M.getNumColumns()];
		Iterator<Object[]> iter = IteratorFactory.getObjectRowIterator(M);
		while( iter.hasNext() ) {
			Object[] tmp = iter.next();
			for( int j=0; j<tmp.length; j++ ) 
				if( collist.contains(j+1) && tmp[j] != null ) {
					if( ret[j] == null )
						ret[j] = new HashMap<>();
					String[] parts = IOUtilFunctions.splitCSV(
							tmp[j].toString(), Lop.DATATYPE_PREFIX);
					ret[j].put(parts[0], Long.parseLong(parts[1]));
				}
		}
		
		return ret;
	}

	private static double[][] generateData(int rows, int cols, HashMap<String,Long>[] RC) {
		double[][] ret = new double[rows][cols];
		for( int i=0; i<rows; i++ ) 
			for( int j=0; j<cols; j++ ) 
				if( RC[j] != null ) {
					ret[i][j] = RC[j].values().toArray(new Long[0])[i%RC[j].size()];
				}
		
		return ret;
	}
}
