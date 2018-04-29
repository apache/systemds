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

package org.apache.sysml.hops.estim;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Basic average case estimator for matrix sparsity:
 * sp = 1 - Math.pow(1-sp1*sp2, k)
 */
public class EstimatorBasicAvg extends SparsityEstimator
{
	@Override
	public double estim(MMNode root) {
		//recursive sparsity evaluation of non-leaf nodes
		double sp1 = !root.getLeft().isLeaf() ? estim(root.getLeft()) :
			OptimizerUtils.getSparsity(root.getLeft().getMatrixCharacteristics());
		double sp2 = !root.getRight().isLeaf() ? estim(root.getRight()) :
			OptimizerUtils.getSparsity(root.getRight().getMatrixCharacteristics());
		return estimIntern(sp1, sp2, root.getRows(), root.getLeft().getCols(), root.getCols());
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1.getMatrixCharacteristics(), m2.getMatrixCharacteristics());
	}

	@Override
	public double estim(MatrixCharacteristics mc1, MatrixCharacteristics mc2) {
		return estimIntern(
			OptimizerUtils.getSparsity(mc1), OptimizerUtils.getSparsity(mc2),
			mc1.getRows(), mc1.getCols(), mc2.getCols());
	}

	private double estimIntern(double sp1, double sp2, long m, long k, long n) {
		return OptimizerUtils.getMatMultSparsity(sp1, sp2, m, k, n, false);
	}
}
