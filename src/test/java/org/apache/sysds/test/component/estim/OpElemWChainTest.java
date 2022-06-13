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

package org.apache.sysds.test.component.estim;

import org.junit.Test;
import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.MMNode;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.apache.commons.lang.NotImplementedException;

/**
 * this is the basic operation check for all estimators with single operations
 */
public class OpElemWChainTest extends AutomatedTestBase 
{
	private final static int m = 1600;
	private final static int n = 700;
	private final static double[] sparsity = new double[]{0.1, 0.04};
	private final static OpCode mult = OpCode.MULT;
	private final static OpCode plus = OpCode.PLUS;

	@Override
	public void setUp() {
		//do  nothing
	}
	//Average Case
	@Test
	public void testAvgMult() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, n, sparsity, mult);
	}
	
	@Test
	public void testAvgPlus() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, n, sparsity, plus);
	}
	
	//Worst Case
	@Test
	public void testWorstMult() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, n, sparsity, mult);
	}
	
	@Test
	public void testWorstPlus() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, n, sparsity, plus);
	}
	
	//DensityMap
	/*@Test
	public void testDMMult() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, n, sparsity, mult);
	}
	
	@Test
	public void testDMPlus() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, n, sparsity, plus);
	}
	
	//MNC
	@Test
	public void testMNCMult() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), m, n, sparsity, mult);
	}
	
	@Test
	public void testMNCPlus() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), m, n, sparsity, plus);
	}*/
	
	//Bitset
	@Test
	public void testBitsetMult() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, n, sparsity, mult);
	}
	
	@Test
	public void testBitsetPlus() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, n, sparsity, plus);
	}
	
	//Layered Graph
	/*@Test
	public void testLGCasemult() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, n, sparsity, mult);
	}
		
	@Test
	public void testLGCaseplus() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, n, sparsity, plus);
	}*/
	
	
	private static void runSparsityEstimateTest(SparsityEstimator estim, int m, int n, double[] sp, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, n, sp[0], 1, 1, "uniform", 3);
		MatrixBlock m2 = MatrixBlock.randOperations(m, n, sp[1], 1, 1, "uniform", 5);
		MatrixBlock m3 = MatrixBlock.randOperations(n, m, sp[1], 1, 1, "uniform", 7);
		MatrixBlock m4 = new MatrixBlock();
		MatrixBlock m5 = new MatrixBlock();
		BinaryOperator bOp;
		double est = 0;
		switch(op) {
			case MULT:
				bOp = new BinaryOperator(Multiply.getMultiplyFnObject());
				m1.binaryOperations(bOp, m2, m4);
				m5 = m4.aggregateBinaryOperations(m4, m3, 
						new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
				est = estim.estim(new MMNode(new MMNode(new MMNode(m1), new MMNode(m2), op), new MMNode(m3), OpCode.MM)).getSparsity();
				// System.out.println(m5.getSparsity());
				// System.out.println(est);
				break;
			case PLUS:
				bOp = new BinaryOperator(Plus.getPlusFnObject());
				m1.binaryOperations(bOp, m2, m4);
				m5 = m4.aggregateBinaryOperations(m4, m3, 
						new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
				est = estim.estim(new MMNode(new MMNode(new MMNode(m1), new MMNode(m2), op), new MMNode(m3), OpCode.MM)).getSparsity();
				// System.out.println(m5.getSparsity());
				// System.out.println(est);
				break;
			default:
				throw new NotImplementedException();
		}
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m5.getSparsity(), (estim instanceof EstimatorBasicWorst) ? 9e-1 : 1e-2);
	}
}
