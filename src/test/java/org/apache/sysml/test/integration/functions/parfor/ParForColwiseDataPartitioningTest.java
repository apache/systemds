/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ParForColwiseDataPartitioningTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "parfor_cdatapartitioning";
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForColwiseDataPartitioningTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 50; 
	private final static int cols1 = (int)Hop.CPThreshold+1;  
	private final static int rows2 = (int)Hop.CPThreshold+1; 
	private final static int cols2 = 50;  
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1d;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
			new String[] { "Rout" })   ); //TODO this specification is not intuitive
	}

	//colwise partitioning
	
	@Test
	public void testParForDataPartitioningNoneLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.NONE, null, false, false);
	}

	@Test
	public void testParForDataPartitioningNoneLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.NONE, PExecMode.LOCAL, false, true);
	}
	
	@Test
	public void testParForDataPartitioningLocalLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.LOCAL, false, false);
	}

	@Test
	public void testParForDataPartitioningLocalLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.LOCAL, false, true);
	}
	
	@Test
	public void testParForDataPartitioningLocalRemoteLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.REMOTE_MR, false, false);
	}

	@Test
	public void testParForDataPartitioningLocalRemoteLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.REMOTE_MR, false, true);
	}

	@Test
	public void testParForDataPartitioningRemoteLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.LOCAL, false, false);
	}

	@Test
	public void testParForDataPartitioningRemoteLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.LOCAL, false, true);
	}
	
	@Test
	public void testParForDataPartitioningRemoteRemoteLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.REMOTE_MR, false, false);
	}

	@Test
	public void testParForDataPartitioningRemoteRemoteLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.REMOTE_MR, false, true);
	}

	@Test
	public void testParForDataPartitioningRemoteSparkLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, false, false);
	}

	@Test
	public void testParForDataPartitioningRemoteSparkLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, false, true);
	}
	
	@Test
	public void testParForDataPartitioningRemoteSparkRemoteLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, false, false);
	}

	@Test
	public void testParForDataPartitioningRemoteSparkRemoteLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, false, true);
	}

	
	//colblockwise partitioning
	

	@Test
	public void testParForDataPartitioningNoneLocalSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.NONE, null, true, false);
	}

	@Test
	public void testParForDataPartitioningNoneLocalSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.NONE, PExecMode.LOCAL, true, true);
	}
	
	@Test
	public void testParForDataPartitioningLocalLocalSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.LOCAL, true, false);
	}

	@Test
	public void testParForDataPartitioningLocalLocalSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.LOCAL, true, true);
	}
	
	@Test
	public void testParForDataPartitioningLocalRemoteSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.REMOTE_MR, true, false);
	}

	@Test
	public void testParForDataPartitioningLocalRemoteSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.REMOTE_MR, true, true);
	}

	@Test
	public void testParForDataPartitioningRemoteLocalSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.LOCAL, true, false);
	}

	@Test
	public void testParForDataPartitioningRemoteLocalSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.LOCAL, true, true);
	}
	
	@Test
	public void testParForDataPartitioningRemoteRemoteSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.REMOTE_MR, true, false);
	}

	@Test
	public void testParForDataPartitioningRemoteRemoteSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.REMOTE_MR, true, true);
	}
	
	@Test
	public void testParForDataPartitioningRemoteSparkLocalSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, true, false);
	}

	@Test
	public void testParForDataPartitioningRemoteSparkLocalSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, true, true);
	}
	
	@Test
	public void testParForDataPartitioningRemoteSparkRemoteSmallDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_MR, true, false);
	}

	@Test
	public void testParForDataPartitioningRemoteSparkRemoteSmallSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_MR, true, true);
	}


	//NOT colwise
	
	@Test
	public void testParForNoDataPartitioningRemoteLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.LOCAL, false, false, true);
	}
	
	@Test
	public void testParForNoDataPartitioningRemoteLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_MR, PExecMode.LOCAL, false, true, true);
	}
	
	@Test
	public void testParForNoDataPartitioningRemoteSparkLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, false, false, true);
	}
	
	@Test
	public void testParForNoDataPartitioningRemoteSparkLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, false, true, true);
	}
	
	/**
	 * 
	 * @param partitioner
	 * @param mode
	 * @param small
	 * @param sparse
	 */
	private void runParForDataPartitioningTest( PDataPartitioner partitioner, PExecMode mode, boolean small, boolean sparse )
	{
		runParForDataPartitioningTest(partitioner, mode, small, sparse, false);
	}
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForDataPartitioningTest( PDataPartitioner partitioner, PExecMode mode, boolean small, boolean sparse, boolean multiParts )
	{
		RUNTIME_PLATFORM oldRT = rtplatform;
		boolean oldUseSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		if( partitioner == PDataPartitioner.REMOTE_SPARK || mode == PExecMode.REMOTE_SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		}

		try
		{
			//inst exec type, influenced via rows
			int rows = small ? rows1 : rows2;
			int cols = small ? cols1 : cols2;
				
			//script
			int scriptNum = -1;
			switch( partitioner )
			{
				case NONE: 
					scriptNum=1; 
					break; 
				case LOCAL: 
					if( mode==PExecMode.LOCAL )
						scriptNum=2; 
					else
						scriptNum=3;
				case REMOTE_MR: 
					if( mode==PExecMode.LOCAL ){
						if( !multiParts )
							scriptNum = 4; 
						else 
							scriptNum = 6;
					}
					else
						scriptNum = 5;	
					break; 
				case REMOTE_SPARK: 
					if( mode==PExecMode.LOCAL ){
						if( !multiParts )
							scriptNum = 7; 
						else 
							scriptNum = 9;
					}
					else
						scriptNum = 8;	
					break; 
				default:
					//do nothing
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + scriptNum + ".dml";
			programArgs = new String[]{"-args", input("V"), 
				Integer.toString(rows), Integer.toString(cols), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + (multiParts?"6":"") + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			long seed = System.nanoTime();
			double sparsity = -1;
			if( sparse )
				sparsity = sparsity2;
			else
				sparsity = sparsity1;
	        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("V", V, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		}
		finally
		{
			rtplatform = oldRT;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseSparkConfig;
		}
	}
}