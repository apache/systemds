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

package org.apache.sysds.test.applications;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * 
 * 
 */
public class ParForSampleTest extends AutomatedTestBase 
{	
	private final static String TEST_NAME = "parfor_sample";
	private final static String TEST_DIR = "applications/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForSampleTest.class.getSimpleName() + "/";
	
	private final static int rows = 2298;
	private final static int cols = 1123;

	private final static double sparsity1 = 0.73;
	private final static double sparsity2 = 0.25;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B1","B2"}) );
	}
	
	@Test
	public void testParForSampleDenseCP() {
		runParForSampleTest(false, ExecType.CP);
	}
	
	@Test
	public void testParForSampleSparseCP() {
		runParForSampleTest(true, ExecType.CP);
	}
	
	@Test
	public void testParForSampleDenseSpark() {
		runParForSampleTest(false, ExecType.SPARK);
	}
	
	@Test
	public void testParForSampleSparseSpark() {
		runParForSampleTest(true, ExecType.SPARK);
	}
		
	@SuppressWarnings({ "unchecked" })
	private void runParForSampleTest( boolean sparse, ExecType et )
	{
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;


		try
		{
			//invocation arguments
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), "0.8 0.2", output("B")};
			
			//generate input data + sequence in first column
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparse?sparsity2:sparsity1, 7); 
			for( int i=0; i<A.length; i++ )
				A[i][0] = (i+1);
			writeInputMatrixWithMTD("A", A, false);
			
			//run test case
			runTest(true, false, null, -1);
			
			//read result data and meta data
			HashMap<CellIndex, Double> B1 = readDMLMatrixFromOutputDir("B1");
			HashMap<CellIndex, Double> B2 = readDMLMatrixFromOutputDir("B2");
			MatrixCharacteristics B1mc = readDMLMetaDataFile("B1");
			MatrixCharacteristics B2mc = readDMLMetaDataFile("B2");
			
			//compare meta data
			Assert.assertEquals(Long.valueOf(rows), Long.valueOf(B1mc.getRows()+B2mc.getRows())); //join full coverage rows
			Assert.assertEquals(Long.valueOf(cols), Long.valueOf(B1mc.getCols())); //full coverage cols
			Assert.assertEquals(Long.valueOf(cols), Long.valueOf(B2mc.getCols())); //full coverage cols
			Assert.assertNotEquals(Long.valueOf(rows), Long.valueOf(B1mc.getRows())); //no sample contains all rows
			Assert.assertNotEquals(Long.valueOf(rows), Long.valueOf(B2mc.getRows())); //no sample contains all rows
			
			//compare data
			HashSet<Integer> probe = new HashSet<>(rows);
			for( int i=0; i<rows; i++ )
				probe.add(i+1);
			for( HashMap<CellIndex, Double> B : new HashMap[]{ B1, B2 } )
				for( Entry<CellIndex,Double> e : B.entrySet() )
					if( e.getKey().column == 1 ) {
						boolean flag = probe.remove(e.getValue().intValue());
						Assert.assertTrue("Wrong return value for "+e.getKey()+": "+e.getValue(), flag);
					}
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
