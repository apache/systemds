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

package org.apache.sysds.test.functions.indexing;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LeftIndexingOp.LeftIndexingMethod;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class LeftIndexingSparseDenseTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_NAME = "LeftIndexingSparseDenseTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + LeftIndexingSparseDenseTest.class.getSimpleName() + "/";
	
	private final static int rows1 = 1073;
	private final static int cols1 = 1050;
	private final static int rows2 = 1073;
	private final static int cols2 = 550;
	
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum LixType {
		LEFT_ALIGNED,
		LEFT2_ALIGNED,
		RIGHT_ALIGNED,
		RIGHT2_ALIGNED,
		CENTERED,
		SINGLE_BLOCK,
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
		
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}
	
	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testSparseMapLeftIndexingLeftAlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.LEFT_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R);
	}
	
	@Test
	public void testSparseMapLeftIndexingLeft2AlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.LEFT2_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R);
	}
	
	@Test
	public void testSparseMapLeftIndexingRightAlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.RIGHT_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R);
	}
	
	@Test
	public void testSparseMapLeftIndexingRight2AlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.RIGHT2_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R);
	}
	
	@Test
	public void testSparseMapLeftIndexingCenteredSP() {
		runLeftIndexingSparseSparseTest(LixType.CENTERED, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R);
	}
	
	@Test
	public void testSparseMapLeftIndexingSingleBlockSP() {
		runLeftIndexingSparseSparseTest(LixType.SINGLE_BLOCK, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R);
	}
	
	// ----
	@Test
	public void testSparseLeftIndexingLeftAlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.LEFT_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX);
	}
	
	@Test
	public void testSparseLeftIndexingLeft2AlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.LEFT2_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX);
	}
	
	@Test
	public void testSparseLeftIndexingRightAlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.RIGHT_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX);
	}
	
	@Test
	public void testSparseLeftIndexingRight2AlignedSP() {
		runLeftIndexingSparseSparseTest(LixType.RIGHT2_ALIGNED, ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX);
	}
	
	@Test
	public void testSparseLeftIndexingCenteredSP() {
		runLeftIndexingSparseSparseTest(LixType.CENTERED, ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX);
	}
	
	@Test
	public void testSparseLeftIndexingSingleBlockSP() {
		runLeftIndexingSparseSparseTest(LixType.SINGLE_BLOCK, ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_L);
	}
	
	public void runLeftIndexingSparseSparseTest(LixType type, ExecType et, LeftIndexingOp.LeftIndexingMethod indexingMethod) 
	{
		int cl = -1;
		int lcols1 = cols1;
		switch( type ){
			case LEFT_ALIGNED: cl = 1; break;
			case LEFT2_ALIGNED: cl = 2; break;
			case RIGHT_ALIGNED: cl = cols1-cols2+1; break;
			case RIGHT2_ALIGNED: cl = cols1-cols2; break;
			case CENTERED: cl = (cols1-cols2)/2; break;
			case SINGLE_BLOCK: cl = 3; lcols1=cols2+7; break;
			
		}
		int cu = cl+cols2-1;
		
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode oldRTP = rtplatform;
		
		//setup range (column lower/upper)
		try {
		    
			if(indexingMethod != null) {
				LeftIndexingOp.FORCED_LEFT_INDEXING = indexingMethod;
			}
			
			if(et == ExecType.SPARK) {
		    	rtplatform = ExecMode.SPARK;
		    }
			else {
				// rtplatform = (et==ExecType.MR)? ExecMode.HADOOP : ExecMode.SINGLE_NODE;
			    rtplatform = ExecMode.HYBRID;
			}
			if( rtplatform == ExecMode.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = type.toString() + "/";
			}

			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), input("B"), 
				String.valueOf(cl), String.valueOf(cu), output("R")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + cl + " " + cu + " " + expectedDir();
			
			//generate input data sets
			double[][] A = getRandomMatrix(rows1, lcols1, -1, 1, sparsity1, 1234);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows2, cols2, -1, 1, sparsity2, 5678);
			writeInputMatrixWithMTD("B", B, true);

			runTest(true, false, null, 6); //2xrblk,2xchk,ix,write
			runRScript(true);
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "DML", "R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(rows1,lcols1,1,1));
		}
		finally {
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			LeftIndexingOp.FORCED_LEFT_INDEXING = null;
		}
	}
}

