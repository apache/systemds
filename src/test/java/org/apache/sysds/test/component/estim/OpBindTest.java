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

import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.EstimatorRowWise;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang3.NotImplementedException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * this is the basic operation check for all estimators with binding operations
 */
@RunWith(value = Parameterized.class)
public class OpBindTest extends AutomatedTestBase
{
	@Parameterized.Parameter(0)
	public int m;
	@Parameterized.Parameter(1)
	public int k;
	@Parameterized.Parameter(2)
	public int n;
	@Parameterized.Parameter(3)
	public double[] sparsity;

	@Override
	public void setUp() {
		//do  nothing
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {m, k, n, sparsity}
			{600, 300, 100, new double[]{0.2, 0.4}},
			{600, 200, 300, new double[]{0.1, 0.15}},
		});
	}

	//Average Case
	@Test
	public void testAvgRbind() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), OpCode.RBIND);
	}
	
	@Test
	public void testAvgCbind() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), OpCode.CBIND);
	}
	
	//Worst Case
	@Test
	public void testWorstRbind() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), OpCode.RBIND);
	}
	
	@Test
	public void testWorstCbind() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), OpCode.CBIND);
	}
	
	//DensityMap
	/*@Test
	public void testDMCaserbind() {
		runSparsityEstimateTest(new EstimatorDensityMap(), OpCode.RBIND);
	}
	
	@Test
	public void testDMCasecbind() {
		runSparsityEstimateTest(new EstimatorDensityMap(), OpCode.CBIND);
	}*/
	
	//MNC
	@Test
	public void testMNCRbind() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), OpCode.RBIND);
	}
		
	@Test
	public void testMNCCbind() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), OpCode.CBIND);
	}

	//Bitset
	@Test
	public void testBitsetCasecbind() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), OpCode.CBIND);
	}
	
	@Test
	public void testBitsetCaserbind() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), OpCode.RBIND);
	}
	
	//Layered Graph
	@Test
	public void testLGCaserbind() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), OpCode.RBIND);
	}
	
	@Test
	public void testLGCasecbind() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), OpCode.CBIND);
	}
		
	//Sample
	/*@Test
	public void testSampleCaserbind() {
		runSparsityEstimateTest(new EstimatorSample(), OpCode.RBIND);
	}
	
	@Test
	public void testSampleCasecbind() {
		runSparsityEstimateTest(new EstimatorSample(), OpCode.CBIND);
	}*/

	// Row Wise Sparsity Estimator
	@Test
	public void testRowWiseRbind() {
		runSparsityEstimateTest(new EstimatorRowWise(), OpCode.RBIND);
	}

	@Test
	public void testRowWiseCbind() {
		runSparsityEstimateTest(new EstimatorRowWise(), OpCode.CBIND);
	}

	
	private void runSparsityEstimateTest(SparsityEstimator estim, OpCode op) {
		MatrixBlock m1;
		MatrixBlock m2;
		MatrixBlock m3 = new MatrixBlock();
		switch(op) {
			case RBIND:
				m1 = MatrixBlock.randOperations(m, k, sparsity[0], 1, 1, "uniform", 3);
				m2 = MatrixBlock.randOperations(n, k, sparsity[1], 1, 1, "uniform", 3);
				m1.append(m2, m3, false);
				break;
			case CBIND:
				m1 = MatrixBlock.randOperations(10, 130, sparsity[0], 1, 1, "uniform", 3);
				m2 = MatrixBlock.randOperations(10, 70, sparsity[1], 1, 1, "uniform", 3);
				m1.append(m2, m3);
				break;
			default:
				throw new NotImplementedException();
		}
		double est = estim.estim(m1, m2, op);
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m3.getSparsity(),
			(estim instanceof EstimatorBasicWorst) ? 5e-1 :
			(estim instanceof EstimatorLayeredGraph) ? 3e-2 : 1e-2);
	}
}
