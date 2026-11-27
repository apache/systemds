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
import org.apache.sysds.runtime.meta.DataCharacteristics;

/**
 * Basic average case estimator for matrix sparsity:
 * sp = 1 - Math.pow(1-sp1*sp2, k)
 */
public class EstimatorBasicAvg extends SparsityEstimator
{
	@Override
	public DataCharacteristics estim(MMNode root) {
		ScalarSparsity leftSparsity = getCachedSparsityInformation(root.getLeft());
		ScalarSparsity rightSparsity = getCachedSparsityInformation(root.getRight());

		ScalarSparsity outSparsity = estimIntern(leftSparsity, rightSparsity, root.getOp());
		root.setSparsityInformation(outSparsity);
		return root.setDataCharacteristics(outSparsity.deriveDataCharacteristics());
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		return estimIntern(m1, m2, op).getAvgSparsity();
	}

	@Override
	public double estim(MatrixBlock m, OpCode op) {
		return estimIntern(m, null, op).getAvgSparsity();
	}

	private ScalarSparsity getCachedSparsityInformation(MMNode node) {
		if(node == null)
			return null;
		// ensure sparsity information is properly cached and reused
		if(node.isLeaf() && node.getSparsityInformation() == null) {
			if(node.getData() != null)
				node.setSparsityInformation(new ScalarSparsity(node.getData()));
			else if(node.getDataCharacteristics() != null)
				node.setSparsityInformation(new ScalarSparsity((int)node.getDataCharacteristics().getRows(),
					(int)node.getDataCharacteristics().getCols(), node.getDataCharacteristics().getSparsity()));
		}
		else if(!node.isLeaf()) {
			estim(node); // recursively obtain sparsity information
		}
		return (ScalarSparsity)node.getSparsityInformation();
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

	private ScalarSparsity estimIntern(MatrixBlock mb1, MatrixBlock mb2, OpCode op) {
		if(mb1 == null)
			return null;
		if(mb2 == null)
			return estimIntern((int)mb1.getNumRows(), (int)mb1.getNumColumns(), mb1.getSparsity(), -1, -1, -1, op);
		return estimIntern((int)mb1.getNumRows(), (int)mb1.getNumColumns(), mb1.getSparsity(),
			(int)mb2.getNumRows(), (int)mb2.getNumColumns(), mb2.getSparsity(), op);
	}

	private ScalarSparsity estimIntern(ScalarSparsity ssp1, ScalarSparsity ssp2, OpCode op) {
		if(ssp1 == null)
			return null;
		if(ssp2 == null)
			return estimIntern((int)ssp1.getNumRows(), (int)ssp1.getNumColumns(), ssp1.getAvgSparsity(), -1, -1, -1, op);
		return estimIntern((int)ssp1.getNumRows(), (int)ssp1.getNumColumns(), ssp1.getAvgSparsity(),
			(int)ssp2.getNumRows(), (int)ssp2.getNumColumns(), ssp2.getAvgSparsity(), op);
	}

	private ScalarSparsity estimIntern(int r1, int c1, double sp1,
		int r2, int c2, double sp2, OpCode op) {
		switch (op) {
			case MM:
				return new ScalarSparsity(r1, c2,
					OptimizerUtils.getMatMultSparsity(sp1, sp2,
					r1, c1, c2, false));
			case MULT:
				return new ScalarSparsity(r1, c1, sp1 * sp2);
			case PLUS:
				return new ScalarSparsity(r1, c1, sp1 + sp2 - sp1 * sp2);
			case EQZERO:
			case DIAG:
			case CBIND:
			case RBIND:
			case NEQZERO:
			case TRANS:
			case RESHAPE:
				return estimExactMetaData(r1, c1, sp1, r2, c2, sp2, op);
			default:
				throw new NotImplementedException();
		}
	}

	public static class ScalarSparsity extends SparsityInformation {
		private double _spar_val;

		public ScalarSparsity(int rlen, int clen, double val) {
			super(rlen, clen);

			_spar_val = val;
		}

		public ScalarSparsity(MatrixBlock mb) {
			super(mb);

			_spar_val = mb.getSparsity();
		}

		@Override
		public long getNnz() {
			return Math.round(_spar_val * getNumRows() * getNumColumns());
		}

		@Override
		public double getAvgSparsity() {
			return _spar_val;
		}
	}
}
