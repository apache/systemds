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
 * sp = Math.min(1, sp1 * k) * Math.min(1, sp2 * k).
 * 
 * Note: for outer-products (i.e., k=1) this worst-case
 * estimate is equivalent to the average case estimate and
 * the exact output sparsity.
 */
public class EstimatorBasicWorst extends SparsityEstimator
{
	@Override
	public double estim(MMNode root) {
		return estim(root, OpCode.MM);
	}


	public double estim(MMNode root, OpCode op) {
		double sp1 = !root.getLeft().isLeaf() ? estim(root.getLeft()) :
			OptimizerUtils.getSparsity(root.getLeft().getMatrixCharacteristics());
		double sp2 = !root.getRight().isLeaf() ? estim(root.getRight()) :
			OptimizerUtils.getSparsity(root.getRight().getMatrixCharacteristics());
		double ret = 0;
		switch (op) {
		case MM:
			ret = estimInternMM(sp1, sp2, root.getRows(), root.getLeft().getCols(), root.getCols());
			root.setSynopsis(ret);
			return ret;
		case MULT:
			ret = sp1 + sp2;
			root.setSynopsis(ret);
			return ret;
		case PLUS:
			ret = Math.min(sp1, sp2);
			root.setSynopsis(ret);
			return ret;
		default:
			throw new NotImplementedException();
		}
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estimIntern(m1, m2, OpCode.MM);
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
			return m1.getSparsity()+m2.getSparsity();
		case PLUS:
			return Math.min(m1.getSparsity(), m2.getSparsity());
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
		return OptimizerUtils.getMatMultSparsity(sp1, sp2, m, k, n, true);
	}
}
