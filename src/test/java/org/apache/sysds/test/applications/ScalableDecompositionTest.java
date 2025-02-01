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

import org.apache.sysds.common.Opcodes;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ScalableDecompositionTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ScalableDecomposition";
	private final static String TEST_DIR = "applications/decomp/";
	private final static String TEST_CLASS_DIR = 
		TEST_DIR + ScalableDecompositionTest.class.getSimpleName() + "/";

	private final static int rows = 1362;
	private final static int cols = 1362;
	private final static int blen = 200;
	private final static double eps = 1e-7;
	
	private enum DecompType {
		CHOLESKY, LU, QR, SOLVE, INVERSE
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C","D","E" })); 
	}

	@Test
	public void testCholeskyCP() {
		runKMeansTest(TEST_NAME1, DecompType.CHOLESKY, false, ExecType.CP);
	}
	
	@Test
	public void testCholeskyRewritesCP() {
		runKMeansTest(TEST_NAME1, DecompType.CHOLESKY, true, ExecType.CP);
	}
	
//	@Test
//	public void testCholeskySP() {
//		runKMeansTest(TEST_NAME1, DecompType.CHOLESKY, false, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testCholeskyRewritesSP() {
//		runKMeansTest(TEST_NAME1, DecompType.CHOLESKY, true, ExecType.SPARK);
//	}
	
//	@Test
//	public void testLUDecompCP() {
//		runKMeansTest(TEST_NAME1, DecompType.LU, false, ExecType.CP);
//	}
//	
//	@Test
//	public void testLUDecompRewritesCP() {
//		runKMeansTest(TEST_NAME1, DecompType.LU, true, ExecType.CP);
//	}
	
//	@Test
//	public void testLUDecompSP() {
//		runKMeansTest(TEST_NAME1, DecompType.LU, false, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testLUDecompRewritesSP() {
//		runKMeansTest(TEST_NAME1, DecompType.LU, true, ExecType.SPARK);
//	}
	
//	@Test
//	public void testQRDecompCP() {
//		runKMeansTest(TEST_NAME1, DecompType.QR, false, ExecType.CP);
//	}
//	
//	@Test
//	public void testQRDecompRewritesCP() {
//		runKMeansTest(TEST_NAME1, DecompType.QR, true, ExecType.CP);
//	}
	
//	@Test
//	public void testQRDecompSP() {
//		runKMeansTest(TEST_NAME1, DecompType.QR, false, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testQRDecompRewritesSP() {
//		runKMeansTest(TEST_NAME1, DecompType.QR, true, ExecType.SPARK);
//	}
	
//	@Test
//	public void testSolveCP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, false, ExecType.CP);
//	}
//	
//	@Test
//	public void testSolveRewritesCP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, true, ExecType.CP);
//	}
	
//	@Test
//	public void testSolveSP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, false, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testSolveRewritesSP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, true, ExecType.SPARK);
//	}
	
//	@Test
//	public void testInverseCP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, false, ExecType.CP);
//	}
//	
//	@Test
//	public void testInverseRewritesCP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, true, ExecType.CP);
//	}
	
//	@Test
//	public void testInverseSP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, false, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testInverseRewritesSP() {
//		runKMeansTest(TEST_NAME1, DecompType.SOLVE, true, ExecType.SPARK);
//	}
	
	
	private void runKMeansTest(String testname, DecompType type, boolean rewrites, ExecType instType)
	{
		boolean oldFlag1 = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean oldFlag2 = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = rewrites;
			
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + testname + ".dml";
			programArgs = new String[]{"-stats", "-explain", "hops", "-args",
				String.valueOf(type.ordinal()), String.valueOf(blen),
				input("A"), input("B"), output("C"), output("D"), output("E") };
			
			switch( type ) {
				case CHOLESKY: {
					MatrixBlock A = MatrixBlock.randOperations(rows, cols, 1.0, -5, 10, "uniform", 7);
					MatrixBlock AtA = A.transposeSelfMatrixMultOperations(new MatrixBlock(), MMTSJType.LEFT);
					writeInputMatrixWithMTD("A", AtA, false);
					runTest(true, false, null, -1);
					HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
					MatrixBlock C2 = LibCommonsMath.unaryOperations(AtA, Opcodes.CHOLESKY.toString());
					TestUtils.compareMatrices(dmlfile, C2, eps);
					break;
				}
				case SOLVE: {
					MatrixBlock A = MatrixBlock.randOperations(rows, cols, 1.0, -5, 10, "uniform", 7);
					MatrixBlock b = MatrixBlock.randOperations(cols, 1, 1.0, -1, 1, "uniform", 3);
					MatrixBlock y = A.aggregateBinaryOperations(A, b, new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
					writeInputMatrixWithMTD("A", A, false);
					writeInputMatrixWithMTD("B", y, false);
					runTest(true, false, null, -1);
					HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
					MatrixBlock C2 = LibCommonsMath.matrixMatrixOperations(A, b, Opcodes.SOLVE.toString());
					TestUtils.compareMatrices(dmlfile, C2, eps);
					break;
				}
				case LU: {
					MatrixBlock A = MatrixBlock.randOperations(rows, cols, 1.0, -5, 10, "uniform", 7);
					writeInputMatrixWithMTD("A", A, false);
					runTest(true, false, null, -1);
					MatrixBlock[] C = LibCommonsMath.multiReturnOperations(A, Opcodes.LU.toString(), 1);
					String[] outputs = new String[]{"C","D","E"};
					for(int i=0; i<outputs.length; i++) {
						HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(outputs[i]);
						TestUtils.compareMatrices(dmlfile, C[i], eps);
					}
					break;
				}
				case QR: {
					MatrixBlock A = MatrixBlock.randOperations(rows, cols, 1.0, -5, 10, "uniform", 7);
					writeInputMatrixWithMTD("A", A, false);
					runTest(true, false, null, -1);
					MatrixBlock[] C = LibCommonsMath.multiReturnOperations(A, Opcodes.QR.toString(), 1);
					String[] outputs = new String[]{"C","D","E"};
					for(int i=0; i<outputs.length; i++) {
						HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(outputs[i]);
						TestUtils.compareMatrices(dmlfile, C[i], eps);
					}
					break;
				}
				case INVERSE: {
					MatrixBlock A = MatrixBlock.randOperations(rows, cols, 1.0, -5, 10, "uniform", 7);
					writeInputMatrixWithMTD("A", A, false);
					runTest(true, false, null, -1);
					HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
					MatrixBlock C2 = LibCommonsMath.unaryOperations(A, Opcodes.INVERSE.toString());
					TestUtils.compareMatrices(dmlfile, C2, eps);
					break;
				}
			}
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag1;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlag2;
		}
	}
}
