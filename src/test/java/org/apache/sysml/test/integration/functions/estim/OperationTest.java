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
import org.apache.sysml.hops.estim.EstimatorBasicAvg;
import org.apache.sysml.hops.estim.EstimatorBasicWorst;
import org.apache.sysml.hops.estim.EstimatorBitsetMM;
import org.apache.sysml.hops.estim.EstimatorDensityMap;
import org.apache.sysml.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysml.hops.estim.EstimatorSample;
import org.apache.sysml.hops.estim.SparsityEstimator;
import org.apache.sysml.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysml.runtime.functionobjects.Equals;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * this is the basic operation check for all estimators with single operations
 */
public class OperationTest extends AutomatedTestBase 
{
	private final static int m = 600;
	private final static int k = 600;
	private final static int n = 600;
	private final static double[] sparsity = new double[]{0.2, 0.4};
	private final static OpCode mult = OpCode.MULT;
	private final static OpCode plus = OpCode.PLUS;
	private final static OpCode rbind = OpCode.RBIND;
	private final static OpCode cbind = OpCode.CBIND;
	private final static OpCode eqzero = OpCode.EQZERO;
	private final static OpCode diag = OpCode.DIAG;
	private final static OpCode neqzero = OpCode.NEQZERO;
	private final static OpCode trans = OpCode.TRANS;
	private final static OpCode reshape = OpCode.RESHAPE;

	@Override
	public void setUp() {
		//do  nothing
	}
	//Average Case
	@Test
	public void testAvgCasemult() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, sparsity, mult);
	}
	
	@Test
	public void testAvgCaseplus() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, sparsity, plus);
	}
	
	@Test
	public void testAvgCaserbind() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, sparsity, rbind);
	}
	
	@Test
	public void testAvgCasecbind() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, sparsity, cbind);
	}
	
	/*@Test
	public void testAvgCaseeqzero() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, 0, sparsity, eqzero);
	}
	
	@Test
	public void testAvgCasediag() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, 0, sparsity, diag);
	}
	
	@Test
	public void testAvgCaseneqzero() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, 0, sparsity, neqzero);
	}
	
	@Test
	public void testAvgCasetrans() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, 0, sparsity, trans);
	}
	
	@Test
	public void testAvgCasereshape() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, 0, sparsity, reshape);
	}*/
	
	//Worst Case
	@Test
	public void testWCasemult() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, sparsity, mult);
	}
	
	@Test
	public void testWCaseplus() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, sparsity, plus);
	}
	
	@Test
	public void testWCaserbind() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, sparsity, rbind);
	}
	
	@Test
	public void testWCasecbind() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, sparsity, cbind);
	}
	
	/*@Test
	public void testWCaseeqzero() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, 0, sparsity, eqzero);
	}
	
	@Test
	public void testWCasediag() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, 0, sparsity, diag);
	}
	
	@Test
	public void testWCaseneqzero() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, 0, sparsity, neqzero);
	}
	
	@Test
	public void testWCasetrans() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, 0, sparsity, trans);
	}
	
	@Test
	public void testWCasereshape() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, 0, sparsity, reshape);
	} */
	
	//DensityMap
	@Test
	public void testDMCasemult() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, sparsity, mult);
	}
	
	@Test
	public void testDMCaseplus() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, sparsity, plus);
	}
	
	/*@Test
	public void testDMCaserbind() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, sparsity, rbind);
	}
	
	@Test
	public void testDMCasecbind() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, sparsity, cbind);
	}
		
	@Test
	public void testDMCaseeqzero() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, eqzero);
	}
	
	@Test
	public void testDMCasediag() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, diag);
	}
		
	@Test
	public void testDMCaseneqzero() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, neqzero);
	}
		
	@Test
	public void testDMCasetrans() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, trans);
	}
		
	@Test
	public void testDMCasereshape() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, reshape);
	}	*/
	
	//MNC
	@Test
	public void testMNCCasemult() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), m, k, n, sparsity, mult);
	}
	
	@Test
	public void testMNCCaseplus() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), m, k, n, sparsity, plus);
	}
		
	@Test
	public void testMNCCaserbind() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), m, k, n, sparsity, rbind);
	}
		
	@Test
	public void testMNCCasecbind() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), m, k, n, sparsity, cbind);
	}
			
	/*@Test
	public void testMNCCaseeqzero() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, eqzero);
	}
		
	@Test
	public void testMNCCasediag() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, diag);
	}
			
	@Test
	public void testMNCCaseneqzero() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, neqzero);
	}
			
	@Test
	public void testMNCCasetrans() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, trans);
	}
			
	@Test
	public void testMNCCasereshape() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, 0, sparsity, reshape);
	}*/
	
	//TODO: Sample, Layered Graph, Bitset
	//TODO: tests for chains
	
	
	private void runSparsityEstimateTest(SparsityEstimator estim, int m, int k, int n, double[] sp, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sp[0], 1, 1, "uniform", 3);
		MatrixBlock m2 = null;
		MatrixBlock m3 = new MatrixBlock();
		BinaryOperator bOp;
		if(n != 0) {
			m2 = MatrixBlock.randOperations(k, n, sp[1], 1, 1, "uniform", 3);
		}
		double est = 0;
		switch(op) {
		case MULT:
			bOp = new BinaryOperator(Multiply.getMultiplyFnObject());
			m1.binaryOperations(bOp, m2, m3);
			est = estim.estim(m1, m2, op);
			System.out.println(est);
			System.out.println(m3.getSparsity());
			break;
		case PLUS:
			bOp = new BinaryOperator(Plus.getPlusFnObject());
			m1.binaryOperations(bOp, m2, m3);
			est = estim.estim(m1, m2, op);
			System.out.println(est);
			System.out.println(m3.getSparsity());
			break;
		case RBIND:
			m1.append(m2, m3, false);
			est = estim.estim(m1, m2, op);
			System.out.println(est);
			System.out.println(m3.getSparsity());
			break;
		case CBIND:
			m1.append(m2, m3);
			est = estim.estim(m1, m2, op);
			System.out.println(est);
			System.out.println(m3.getSparsity());
			break;
		case EQZERO:
			bOp = new BinaryOperator(Equals.getEqualsFnObject());
			m1.binaryOperationsInPlace(bOp, m3);
			est = estim.estim(m1, op);
			break;
		case DIAG:
			break;
		case NEQZERO:
			break;
		case TRANS:
			break;
		case RESHAPE:
			break;
		//TODO: implement missing tests
		}
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m3.getSparsity(), (estim instanceof EstimatorBasicWorst) ? 5e-1 : 1e-2);
	}
}
