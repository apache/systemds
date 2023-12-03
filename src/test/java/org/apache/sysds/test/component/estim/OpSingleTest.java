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

import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.junit.Test;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * this is the basic operation check for all estimators with single operations
 */
public class OpSingleTest extends AutomatedTestBase 
{
	private final static int m = 600;
	private final static int k = 300;
	private final static double sparsity = 0.2;
//	private final static OpCode eqzero = OpCode.EQZERO;
	private final static OpCode diag = OpCode.DIAG;
	private final static OpCode neqzero = OpCode.NEQZERO;
	private final static OpCode trans = OpCode.TRANS;
	private final static OpCode reshape = OpCode.RESHAPE;

	@Override
	public void setUp() {
		//do  nothing
	}
	
	//Average Case
//	@Test
//	public void testAvgEqzero() {
//		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, eqzero);
//	}
	
//	@Test
//	public void testAvgDiag() {
//		runSparsityEstimateTest(new EstimatorBasicAvg(), m, m, sparsity, diag);
//	}
	
	@Test
	public void testAvgNeqzero() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testAvgTrans() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, trans);
	}
	
	@Test
	public void testAvgReshape() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, reshape);
	}
	
	//Worst Case
//	@Test
//	public void testWorstEqzero() {
//		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, eqzero);
//	}
	
//	@Test
//	public void testWCasediag() {
//		runSparsityEstimateTest(new EstimatorBasicWorst(), m, m, sparsity, diag);
//	}
	
	@Test
	public void testWorstNeqzero() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testWoestTrans() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, trans);
	}
	
	@Test
	public void testWorstReshape() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, reshape);
	} 
	
//	//DensityMap
//	@Test
//	public void testDMCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, eqzero);
//	}
//	
//	@Test
//	public void testDMCasediag() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, m, sparsity, diag);
//	}
//	
//	@Test
//	public void testDMCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, neqzero);
//	}
//	
//	@Test
//	public void testDMCasetrans() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, trans);
//	}
//		
//	@Test
//	public void testDMCasereshape() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, reshape);
//	}
//	
//	//MNC
//	@Test
//	public void testMNCCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, eqzero);
//	}
//	
//	@Test
//	public void testMNCCasediag() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, m, sparsity, diag);
//	}
//	
//	@Test
//	public void testMNCCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, neqzero);
//	}
//	
//	@Test
//	public void testMNCCasetrans() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, trans);
//	}
//	
//	@Test
//	public void testMNCCasereshape() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, reshape);
//	}
//	
	//Bitset
//	@Test
//	public void testBitsetCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, eqzero);
//	}
	
//	@Test
//	public void testBitsetCasediag() {
//		runSparsityEstimateTest(new EstimatorBitsetMM(), m, m, sparsity, diag);
//	}
	
	@Test
	public void testBitsetNeqzero() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testBitsetTrans() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, trans);
	}
	
	@Test
	public void testBitsetReshape() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, reshape);
	}
	
//	//Layered Graph
//	@Test
//	public void testLGCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, sparsity, eqzero);
//	}
//	
	@Test
	public void testLGCasediagM() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, m, sparsity, diag);
	}

	@Test
	public void testLGCasediagV() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, 1, sparsity, diag);
	}
//	
//	@Test
//	public void testLGCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, sparsity, neqzero);
//	}
//	
	@Test
	public void testLGCasetrans() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, sparsity, trans);
	}

//	@Test
//	public void testLGCasereshape() {
//		runSparsityEstimateTest(new EstimatorLayeredGraph(), m, k, sparsity, reshape);
//	}
//	
//	//Sample
//	@Test
//	public void testSampleCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorSample(), m, k, sparsity, eqzero);
//	}
//	
//	@Test
//	public void testSampleCasediag() {
//		runSparsityEstimateTest(new EstimatorSample(), m, m, sparsity, diag);
//	}
//	
//	@Test
//	public void testSampleCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorSample(), m, k, sparsity, neqzero);
//	}
//	
//	@Test
//	public void testSampleCasetrans() {
//		runSparsityEstimateTest(new EstimatorSample(), m, k, sparsity, trans);
//	}
//	
//	@Test
//	public void testSampleCasereshape() {
//		runSparsityEstimateTest(new EstimatorSample(), m, k, sparsity, reshape);
//	}
	
	private static void runSparsityEstimateTest(SparsityEstimator estim, int m, int k, double sp, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sp, 1, 1, "uniform", 3);
		MatrixBlock m2 = new MatrixBlock();
		double est = 0;
		switch(op) {
			case EQZERO:
				//TODO find out how to do eqzero
			case DIAG:
				m2 = m1.getNumColumns() == 1
						? LibMatrixReorg.diag(m1, new MatrixBlock(m1.getNumRows(), m1.getNumRows(), false))
						: LibMatrixReorg.diag(m1, new MatrixBlock(m1.getNumRows(), 1, false));
				est = estim.estim(m1, op);
				break;
			case NEQZERO:
				m2 = m1;
				est = estim.estim(m1, op);
				break;
			case TRANS:
				m2 = m1;
				est = estim.estim(m1, op);
				break;
			case RESHAPE:
				m2 = m1;
				est = estim.estim(m1, op);
				break;
			default:
				throw new NotImplementedException();
		}
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m2.getSparsity(),
			(estim instanceof EstimatorBasicWorst) ? 5e-1 :
			(estim instanceof EstimatorLayeredGraph) ? 3e-2 : 2e-2);
	}
}
