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

package org.apache.sysds.test.functions.reorg;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;


public class FullReverseTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Reverse1";
	private final static String TEST_NAME2 = "Reverse2";
	
	private final static String TEST_DIR = "functions/reorg/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullReverseTest.class.getSimpleName() + "/";
	
	//single-threaded execution
	private final static int rows1 = 201;
	private final static int cols1 = 100;
	//multi-threaded / distributed execution
	private final static int rows2 = 2017;
	private final static int cols2 = 1001;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"B"}));
	}

	@Test
	public void testReverseVectorDenseCP() {
		runReverseTest(TEST_NAME1, false, rows1, 1, ExecType.CP);
	}
	
	@Test
	public void testReverseVectorSparseCP() {
		runReverseTest(TEST_NAME1, true, rows1, 1, ExecType.CP);
	}

	@Test
	public void testReverseVectorDenseCPMultiThread() {
		runReverseTest(TEST_NAME1, false, rows2, 1, ExecType.CP);
	}

	@Test
	public void testReverseVectorSparseCPMultiThread() {
		runReverseTest(TEST_NAME1, true, rows2, 1, ExecType.CP);
	}

	@Test
	public void testReverseVectorDenseSP() {
		runReverseTest(TEST_NAME1, false, rows2, 1, ExecType.SPARK);
	}
	
	@Test
	public void testReverseVectorSparseSP() {
		runReverseTest(TEST_NAME1, true, rows2, 1, ExecType.SPARK);
	}
	
	@Test
	public void testReverseMatrixDenseCP() {
		runReverseTest(TEST_NAME1, false, rows1, cols1, ExecType.CP);
	}
	
	@Test
	public void testReverseMatrixSparseCP() {
		runReverseTest(TEST_NAME1, true, rows1, cols1, ExecType.CP);
	}
	
	@Test
	public void testReverseMatrixDenseSP() {
		runReverseTest(TEST_NAME1, false, rows2, cols2, ExecType.SPARK);
	}
	
	@Test
	public void testReverseMatrixSparseSP() {
		runReverseTest(TEST_NAME1, true, rows2, cols2, ExecType.SPARK);
	}	

	@Test
	public void testReverseVectorDenseRewriteCP() {
		runReverseTest(TEST_NAME2, false, rows1, 1, ExecType.CP);
	}
	
	@Test
	public void testReverseMatrixDenseRewriteCP() {
		runReverseTest(TEST_NAME2, false, rows1, 1, ExecType.CP);
	}
	
	private void runReverseTest(String testname, boolean sparse, int rows, int cols, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		String TEST_NAME = testname;
		
		try
		{
			double sparsity = sparse ? sparsity2 : sparsity1;
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-explain","-args", input("A"), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check generated opcode
			if( instType == ExecType.CP )
				Assert.assertTrue("Missing opcode: rev", Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.REV.toString()));
			else if ( instType == ExecType.SPARK )
				Assert.assertTrue("Missing opcode: "+Instruction.SP_INST_PREFIX+Opcodes.REV.toString(), Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+Opcodes.REV));
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
