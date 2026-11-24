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

package org.apache.sysds.hops.estim;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.estim.EstimatorBasicAvg.ScalarSparsity;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Basic average case estimator for matrix sparsity:
 * sp = Math.min(1, sp1 * k) * Math.min(1, sp2 * k).
 * 
 * Note: for outer-products (i.e., k=1) this worst-case
 * estimate is equivalent to the average case estimate and
 * the exact output sparsity.
 */
public class EstimatorBasicWorst extends SparsityEstimator {
	@Override
	public ScalarSparsity estim(MMNode root) {
		if (!root.getLeft().isLeaf())
			estim(root.getLeft()); // obtain sparsity information
		if (root.getRight()!=null && !root.getRight().isLeaf())
			estim(root.getRight()); // obtain sparsity information
		SparsityInformation spInf1 = !root.getLeft().isLeaf() ?
			estim(root.getLeft()) : root.getLeft().getSparsityInformation();
		SparsityInformation spInf2 = root.getRight()==null ? null :
			!root.getRight().isLeaf() ? estim(root.getRight()) : 
			root.getRight().getSparsityInformation();
		ScalarSparsity spEst = estimIntern(spInf1, spInf2, root.getOp());
		root.setSparsityInformation(spEst);
		return spEst;
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		return estimIntern(new ScalarSparsity(m1), new ScalarSparsity(m2), op).getAvgSparsity();
	}

	@Override
	public double estim(MatrixBlock m, OpCode op) {
		return estimIntern(new ScalarSparsity(m), null, op).getAvgSparsity();
	}

	protected void obtainSparsityInformation(MMNode node) {
		MatrixBlock mb = node.getData();
		if(mb != null && mb.getDataCharacteristics().getSparsity() == -1) {
			if(mb.getNonZeros() != 0) {
				mb.getDataCharacteristics().setNonZeros(mb.getNonZeros());
			}
			else {
				mb.recomputeNonZeros();
				mb.getDataCharacteristics().setNonZeros(mb.getNonZeros());
			}
		}
	}

	@Override
	public MMNode getInputMMNode(MatrixBlock mb) {
		MMNode node = new MMNode(mb);
		obtainSparsityInformation(node);
		return node;
	}

	private ScalarSparsity estimIntern(SparsityInformation spInf1, SparsityInformation spInf2, OpCode op) {
		switch (op) {
			case MM:
				return new ScalarSparsity(spInf1.getNumRows(), spInf2.getNumColumns(),
					OptimizerUtils.getMatMultNnz(spInf1.getAvgSparsity(), spInf2.getAvgSparsity(),
					spInf1.getNumRows(), spInf1.getNumColumns(), spInf2.getNumColumns(), true));
			case MULT:
				return new ScalarSparsity(spInf1.getNumRows(), spInf1.getNumColumns(),
					OptimizerUtils.getNnz(spInf1.getNumRows(), spInf1.getNumColumns(),
						Math.min(spInf1.getAvgSparsity(), spInf2.getAvgSparsity())));
			case PLUS:
				return new ScalarSparsity(spInf1.getNumRows(), spInf1.getNumColumns(),
					OptimizerUtils.getNnz(spInf1.getNumRows(), spInf1.getNumColumns(), 
						Math.min(spInf1.getAvgSparsity() + spInf2.getAvgSparsity(), 1)));
			case EQZERO:
			case DIAG:
			case CBIND:
			case RBIND:
			case NEQZERO:
			case TRANS:
			case RESHAPE:
				return estimExactMetaData(spInf1, spInf2, op);
			default:
				throw new NotImplementedException();
		}
	}
}
