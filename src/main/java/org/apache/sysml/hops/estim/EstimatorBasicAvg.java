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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Basic average case estimator for matrix sparsity:
 * sp = 1 - Math.pow(1-sp1*sp2, k)
 */
public class EstimatorBasicAvg extends SparsityEstimator {
	
	@Override
	public double estim(MMNode root) {
		return estim(root, OpCode.MM);
	}


	public double estim(MMNode root, OpCode op) {
		double sp1 = !root.getLeft().isLeaf() ? estim(root.getLeft(), root.getLeft().getOp()) :
			OptimizerUtils.getSparsity(root.getLeft().getMatrixCharacteristics());
		double sp2 = !root.getRight().isLeaf() ? estim(root.getRight(), root.getRight().getOp()) :
			OptimizerUtils.getSparsity(root.getRight().getMatrixCharacteristics());
		MatrixBlock m1 = root.getLeft().getData();
		MatrixBlock m2 = root.getRight().getData();
		double ret = 0;
		switch (op) {
		case MM:
			ret = estimInternMM(sp1, sp2, root.getRows(), root.getLeft().getCols(), root.getCols());
			root.setSynopsis(ret);
			return ret;
		case MULT:
			ret = sp1 + sp2 - sp1 * sp2;
			root.setSynopsis(ret);
			return ret;
		case PLUS:
			ret = sp1 * sp2;
			root.setSynopsis(ret);
			return ret;
		case CBIND:
			return OptimizerUtils.getSparsity(m1.getNumRows(),
				m1.getNumColumns() + m1.getNumColumns(), m1.getNonZeros() + m2.getNonZeros());
		case RBIND:
			return OptimizerUtils.getSparsity(m1.getNumRows() + m2.getNumRows(),
				m1.getNumColumns(), m1.getNonZeros() + m2.getNonZeros());
		default:
			throw new NotImplementedException();
		}
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estimInternMM(m1.getSparsity(), m2.getSparsity(),
			m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns());
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		return estimIntern(m1, m2, op);
	}

	@Override
	public double estim(MatrixBlock m, OpCode op) {
		return estimIntern(m, null, op);
	}

	private double estimIntern(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		switch (op) {
			case MM:
				return estimInternMM(m1.getSparsity(), m2.getSparsity(),
					m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns());
			case MULT:
				return m1.getSparsity() * m2.getSparsity();
			case PLUS:
				return m1.getSparsity() + m2.getSparsity() - m1.getSparsity() * m2.getSparsity();
			case EQZERO:
				return OptimizerUtils.getSparsity(m1.getNumRows(), m1.getNumColumns(),
					(long) m1.getNumRows() * m1.getNumColumns() - m1.getNonZeros());
			case DIAG:
				return (m1.getNumColumns() == 1) ?
					OptimizerUtils.getSparsity(m1.getNumRows(), m1.getNumRows(), m1.getNonZeros()) :
					OptimizerUtils.getSparsity(m1.getNumRows(), 1, Math.min(m1.getNumRows(), m1.getNonZeros()));
			// binary operations that preserve sparsity exactly
			case CBIND:
				return OptimizerUtils.getSparsity(m1.getNumRows(),
					m1.getNumColumns() + m1.getNumColumns(), m1.getNonZeros() + m2.getNonZeros());
			case RBIND:
				return OptimizerUtils.getSparsity(m1.getNumRows() + m2.getNumRows(),
					m1.getNumColumns(), m1.getNonZeros() + m2.getNonZeros());
			// unary operation that preserve sparsity exactly
			case NEQZERO:
				return m1.getSparsity();
			case TRANS:
				return m1.getSparsity();
			case RESHAPE:
				return m1.getSparsity();
			default:
				throw new NotImplementedException();
		}
	}

	private double estimInternMM(double sp1, double sp2, long m, long k, long n) {
		return OptimizerUtils.getMatMultSparsity(sp1, sp2, m, k, n, false);
	}
}
