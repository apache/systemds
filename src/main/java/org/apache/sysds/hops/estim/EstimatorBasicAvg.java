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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Basic average case estimator for matrix sparsity:
 * sp = 1 - Math.pow(1-sp1*sp2, k)
 */
public class EstimatorBasicAvg extends SparsityEstimator
{
	@Override
	public ScalarSparsity estim(MMNode root) {
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
				return new ScalarSparsity((int)spInf1.getNumRows(), (int)spInf2.getNumColumns(),
					OptimizerUtils.getMatMultNnz(spInf1.getAvgSparsity(), spInf2.getAvgSparsity(),
					spInf1.getNumRows(), spInf1.getNumColumns(), spInf2.getNumColumns(), false));
			case MULT:
				return new ScalarSparsity((int)spInf1.getNumRows(), (int)spInf1.getNumColumns(),
					OptimizerUtils.getNnz(spInf1.getNumRows(), spInf1.getNumColumns(),
						spInf1.getAvgSparsity() * spInf2.getAvgSparsity()));
			case PLUS:
				return new ScalarSparsity((int)spInf1.getNumRows(), (int)spInf1.getNumColumns(),
					OptimizerUtils.getNnz(spInf1.getNumRows(), spInf1.getNumColumns(),
						spInf1.getAvgSparsity() + spInf2.getAvgSparsity() - spInf1.getAvgSparsity() * spInf2.getAvgSparsity()));
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

	public static class ScalarSparsity extends SparsityInformation {
		private final double _sparsityValue;

		public ScalarSparsity(int nRows, int nCols, double sparsity) {
			super(nRows, nCols);
			_sparsityValue = sparsity;
		}

		public ScalarSparsity(int nRows, int nCols, long nnz) {
			this(nRows, nCols, (double)nnz / (nRows * nCols));
		}

		public ScalarSparsity(MatrixBlock mb) {
			super(mb);
			if(mb.getSparsity() < 0)
				mb.recomputeNonZeros();
			_sparsityValue = mb.getSparsity();
		}

		@Override
		public long getNonZeros() {
			return Double.valueOf(getAvgSparsity() * getNumRows() * getNumColumns())
				.longValue();
		}

		@Override
		public double getAvgSparsity() {
			return _sparsityValue;
		}
	}
}
