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

package org.apache.sysml.test.integration.functions.estim;

import org.junit.Test;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.commons.lang.NotImplementedException;
import org.apache.sysml.hops.estim.EstimatorBasicAvg;
import org.apache.sysml.hops.estim.EstimatorBasicWorst;
import org.apache.sysml.hops.estim.EstimatorDensityMap;
import org.apache.sysml.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysml.hops.estim.SparsityEstimator;
import org.apache.sysml.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * this is the basic operation check for all estimators with single operations
 */
public class OpElemWTest extends AutomatedTestBase 
{
	//TODO experiment with m>2n for MNC (currently suboptimal accuracy)
	private final static int m = 600;
	private final static int n = 700;
	private final static double[] sparsity = new double[]{0.1, 0.04};
	private final static OpCode mult = OpCode.MULT;
	private final static OpCode plus = OpCode.PLUS;
//	private final static OpCode rbind = OpCode.RBIND;
//	private final static OpCode cbind = OpCode.CBIND;
//	private final static OpCode eqzero = OpCode.EQZERO;
//	private final static OpCode diag = OpCode.DIAG;
//	private final static OpCode neqzero = OpCode.NEQZERO;
//	private final static OpCode trans = OpCode.TRANS;
//	private final static OpCode reshape = OpCode.RESHAPE;

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
	@Test
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
	}
	
	//Bitset
	/*@Test
	public void testBitsetCasemult() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, n, sparsity, mult);
	}
	
	@Test
	public void testBitsetCaseplus() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, n, sparsity, plus);
	}
	
	//Layered Graph
	@Test
	public void testLGCasemult() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, n, sparsity, mult);
	}
		
	@Test
	public void testLGCaseplus() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, n, sparsity, plus);
	}
	
	//Sample
	@Test
	public void testSampleCasemult() {
		runSparsityEstimateTest(new EstimatorSample(), m, k, n, sparsity, mult);
	}
		
	@Test
	public void testSampleCaseplus() {
		runSparsityEstimateTest(new EstimatorSample(), m, k, n, sparsity, plus);
	}*/
	
	
	private void runSparsityEstimateTest(SparsityEstimator estim, int m, int n, double[] sp, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, n, sp[0], 1, 1, "uniform", 3);
		MatrixBlock m2 = MatrixBlock.randOperations(m, n, sp[1], 1, 1, "uniform", 3);
		MatrixBlock m3 = new MatrixBlock();
		BinaryOperator bOp;
		double est = 0;
		switch(op) {
			case MULT:
				bOp = new BinaryOperator(Multiply.getMultiplyFnObject());
				m1.binaryOperations(bOp, m2, m3);
				est = estim.estim(m1, m2, op);
				break;
			case PLUS:
				bOp = new BinaryOperator(Plus.getPlusFnObject());
				m1.binaryOperations(bOp, m2, m3);
				est = estim.estim(m1, m2, op);
				break;
			default:
				throw new NotImplementedException();
		}
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m3.getSparsity(), (estim instanceof EstimatorBasicWorst) ? 5e-1 : 1e-3);
	}
}
