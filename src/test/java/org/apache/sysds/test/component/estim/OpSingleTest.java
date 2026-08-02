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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.EstimatorRowWise;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

/**
 * this is the basic operation check for all estimators with single operations
 */
@RunWith(value = Parameterized.class)
public class OpSingleTest extends AutomatedTestBase 
{
	@Parameterized.Parameter(0)
	public int m;
	@Parameterized.Parameter(1)
	public int k_param;
	@Parameterized.Parameter(2)
	public double sparsity;

	@Override
	public void setUp() {
		//do  nothing
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {m, k_param, sparsity}
			{600, 300, 0.2},
			{200, 1200, 0.6},
		});
	}

	//Average Case
//	@Test
//	public void testAvgEqzero() {
//		runSparsityEstimateTest(new EstimatorBasicAvg(), k_param, OpCode.EQZERO);
//	}
	
//	@Test
//	public void testAvgDiag() {
//		runSparsityEstimateTest(new EstimatorBasicAvg(), m, OpCode.DIAG);
//	}
	
	@Test
	public void testAvgNeqzero() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), k_param, OpCode.NEQZERO);
	}
	
	@Test
	public void testAvgTrans() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), k_param, OpCode.TRANS);
	}
	
	@Test
	public void testAvgReshape() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), k_param, OpCode.RESHAPE);
	}
	
	//Worst Case
//	@Test
//	public void testWorstEqzero() {
//		runSparsityEstimateTest(new EstimatorBasicWorst(), k_param, OpCode.EQZERO);
//	}
	
//	@Test
//	public void testWCasediag() {
//		runSparsityEstimateTest(new EstimatorBasicWorst(), m, OpCode.DIAG);
//	}
	
	@Test
	public void testWorstNeqzero() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), k_param, OpCode.NEQZERO);
	}
	
	@Test
	public void testWoestTrans() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), k_param, OpCode.TRANS);
	}
	
	@Test
	public void testWorstReshape() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), k_param, OpCode.RESHAPE);
	} 
	
//	//DensityMap
//	@Test
//	public void testDMCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.EQZERO);
//	}
//	
//	@Test
//	public void testDMCasediag() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, OpCode.DIAG);
//	}
//	
//	@Test
//	public void testDMCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.NEQZERO);
//	}
//	
//	@Test
//	public void testDMCasetrans() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.TRANS);
//	}
//		
//	@Test
//	public void testDMCasereshape() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.RESHAPE);
//	}
//	
//	//MNC
//	@Test
//	public void testMNCCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.EQZERO);
//	}
//	
//	@Test
//	public void testMNCCasediag() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), m, OpCode.DIAG);
//	}
//	
//	@Test
//	public void testMNCCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.NEQZERO);
//	}
//	
//	@Test
//	public void testMNCCasetrans() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.TRANS);
//	}
//	
//	@Test
//	public void testMNCCasereshape() {
//		runSparsityEstimateTest(new EstimatorDensityMap(), k_param, OpCode.RESHAPE);
//	}
//	
	//Bitset
//	@Test
//	public void testBitsetCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorBitsetMM(), k_param, OpCode.EQZERO);
//	}
	
//	@Test
//	public void testBitsetCasediag() {
//		runSparsityEstimateTest(new EstimatorBitsetMM(), m, OpCode.DIAG);
//	}
	
	@Test
	public void testBitsetNeqzero() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), k_param, OpCode.NEQZERO);
	}
	
	@Test
	public void testBitsetTrans() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), k_param, OpCode.TRANS);
	}
	
	@Test
	public void testBitsetReshape() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), k_param, OpCode.RESHAPE);
	}
	
//	//Layered Graph
//	@Test
//	public void testLGCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13), k_param, OpCode.EQZERO);
//	}
//	
	@Test
	public void testLGCasediagM() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13), m, OpCode.DIAG);
	}

	@Test
	public void testLGCasediagV() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13), 1, OpCode.DIAG);
	}
//	
//	@Test
//	public void testLGCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13), k_param, OpCode.NEQZERO);
//	}
//	
	@Test
	public void testLGCasetrans() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13), k_param, OpCode.TRANS);
	}

//	@Test
//	public void testLGCasereshape() {
//		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13), k_param, OpCode.RESHAPE);
//	}
//	
//	//Sample
//	@Test
//	public void testSampleCaseeqzero() {
//		runSparsityEstimateTest(new EstimatorSample(), k_param, OpCode.EQZERO);
//	}
//	
//	@Test
//	public void testSampleCasediag() {
//		runSparsityEstimateTest(new EstimatorSample(), m, OpCode.DIAG);
//	}
//	
//	@Test
//	public void testSampleCaseneqzero() {
//		runSparsityEstimateTest(new EstimatorSample(), k_param, OpCode.NEQZERO);
//	}
//	
//	@Test
//	public void testSampleCasetrans() {
//		runSparsityEstimateTest(new EstimatorSample(), k_param, OpCode.TRANS);
//	}
//	
//	@Test
//	public void testSampleCasereshape() {
//		runSparsityEstimateTest(new EstimatorSample(), k_param, OpCode.RESHAPE);
//	}

	// Row Wise Sparsity Estimator
	@Test
	public void testRowWiseEqzero() {
		runSparsityEstimateTest(new EstimatorRowWise(), k_param, OpCode.EQZERO);
	}

	@Test
	public void testRowWiseDiagM() {
		runSparsityEstimateTest(new EstimatorRowWise(), m, OpCode.DIAG);
	}

	@Test
	public void testRowWiseDiagV() {
		runSparsityEstimateTest(new EstimatorRowWise(), 1, OpCode.DIAG);
	}

	@Test
	public void testRowWiseNeqzero() {
		runSparsityEstimateTest(new EstimatorRowWise(), k_param, OpCode.NEQZERO);
	}

	@Test
	public void testRowWiseTrans() {
		runSparsityEstimateTest(new EstimatorRowWise(), k_param, OpCode.TRANS);
	}

	@Test
	public void testRowWiseReshape() {
		runSparsityEstimateTest(new EstimatorRowWise(), k_param, OpCode.RESHAPE);
	}

	private void runSparsityEstimateTest(SparsityEstimator estim, int k, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sparsity, 1, 1, "uniform", 3);
		MatrixBlock m2;
		double ref = -1;
		switch(op) {
			case EQZERO:
				ref = 1 - m1.getSparsity();
				break;
			case DIAG:
				m2 = m1.getNumColumns() == 1
						? LibMatrixReorg.diag(m1, new MatrixBlock(m1.getNumRows(), m1.getNumRows(), false))
						: LibMatrixReorg.diag(m1, new MatrixBlock(m1.getNumRows(), 1, false));
				ref = m2.getSparsity();
				break;
			case NEQZERO:
			case TRANS:
			case RESHAPE:
				m2 = m1;
				ref = m2.getSparsity();
				break;
			default:
				throw new NotImplementedException();
		}
		double est = estim.estim(m1, op);
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, ref,
			(estim instanceof EstimatorBasicWorst) ? 5e-1 :
			(estim instanceof EstimatorLayeredGraph) ? 3e-2 : 2e-2);
	}
}
