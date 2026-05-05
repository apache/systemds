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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * This estimator implements an approach based on row-wise sparsity estimation,
 * introduced in
 * Lin, Chunxu, Wensheng Luo, Yixiang Fang, Chenhao Ma, Xilin Liu and Yuchi Ma:
 * On Efficient Large Sparse Matrix Chain Multiplication.
 * Proceedings of the ACM on Management of Data 2 (2024): 1 - 27.
 */
public class EstimatorRowWise extends SparsityEstimator {
	@Override
	public DataCharacteristics estim(MMNode root) {
		estimInternChain(root);
		double sparsity = ((RSVector)root.getSynopsis()).avg();

		DataCharacteristics outputCharacteristics = deriveOutputCharacteristics(root, sparsity);
		return root.setDataCharacteristics(outputCharacteristics);
	}

	@Override 
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( isExactMetadataOp(op) ) {
			return estimExactMetaData(m1.getDataCharacteristics(),
				m2.getDataCharacteristics(), op).getSparsity();
		}

		RSVector rsOut = estimIntern(m1, m2, op);
		return rsOut.avg();
	}

	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getDataCharacteristics(), null, op).getSparsity();
		throw new NotImplementedException();
	}

	private void estimInternChain(MMNode node) {
		estimInternChain(node, null, null);
	}

	private void estimInternChain(MMNode node, RSVector rsRightNeighbor, OpCode opRightNeighbor) {
		RSVector rsOut;
		if(node.isLeaf()) {
			MatrixBlock mb = node.getData();
			if(rsRightNeighbor != null)
				rsOut = estimIntern(mb, rsRightNeighbor, opRightNeighbor);
			else
				rsOut = getRowWiseSparsityVector(mb);
		}
		else {
			switch(node.getOp()) {
				case MM:
					estimInternChain(node.getRight(), rsRightNeighbor, opRightNeighbor);
					estimInternChain(node.getLeft(), (RSVector)(node.getRight().getSynopsis()), node.getOp());
					rsOut = (RSVector)node.getLeft().getSynopsis();
					break;
				case CBIND:
					/** NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into a cbind operation when having only row sparsity vectors
					 */
					estimInternChain(node.getLeft());
					estimInternChain(node.getRight());
					RSVector rsCBind = estimInternCBind((RSVector)(node.getLeft().getSynopsis()), (RSVector)(node.getRight().getSynopsis()));
					if(rsRightNeighbor != null)
						rsOut = (RSVector)estimIntern(rsCBind, rsRightNeighbor, opRightNeighbor);
					else
						rsOut = (RSVector)rsCBind;
					break;
				case RBIND:
					/** NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an rbind operation when having only row sparsity vectors
					 */
					estimInternChain(node.getLeft());
					estimInternChain(node.getRight());
					RSVector rsRBind = estimInternRBind((RSVector)(node.getLeft().getSynopsis()), (RSVector)(node.getRight().getSynopsis()));
					if(rsRightNeighbor != null)
						rsOut = (RSVector)estimIntern(rsRBind, rsRightNeighbor, opRightNeighbor);
					else
						rsOut = (RSVector)rsRBind;
					break;
				default:
					throw new NotImplementedException("Chain estimation for operator " + node.getOp().toString() +
					" is not supported yet.");
			}
		}
		node.setSynopsis(rsOut);
		node.setDataCharacteristics(deriveOutputCharacteristics(node, rsOut.avg()));
		return;
	}

	private RSVector estimIntern(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		RSVector rsM2 = getRowWiseSparsityVector(m2);
		return estimIntern(m1, rsM2, op);
	}

	private RSVector estimIntern(MatrixBlock m1, RSVector rsM2, OpCode op) {
		switch(op) {
			case MM:
				return estimInternMM(m1, rsM2);
			case CBIND:
				return estimInternCBind(getRowWiseSparsityVector(m1), rsM2);
			case RBIND:
				return estimInternRBind(getRowWiseSparsityVector(m1), rsM2);
			default:
				throw new NotImplementedException("Sparsity estimation for operation " + op.toString() + " not supported yet.");
		}
	}

	private RSVector estimIntern(RSVector rsM1, RSVector rsM2, OpCode op) {
		switch(op) {
			case MM:
				return estimInternMM(rsM1, rsM2);
			// case CBIND:
			// 	return estimInternCBind(rsM1, rsM2);
			// case RBIND:
			// 	return estimInternRBind(rsM1, rsM2);
			default:
				throw new NotImplementedException("Sparsity estimation for operation " + op.toString() + " not supported yet.");
		}
	}

	// Corresponds to Algorithm 1 in the publication
	private RSVector estimInternMM(MatrixBlock m1, RSVector rsM2) {
		RSVector rsOut = new RSVector(IntStream.range(0, m1.getNumRows()).mapToDouble(
			r -> (double) 1 - IntStream.of(getNonZeroColumnIndices(m1, r)).mapToDouble(
					c -> (double) 1 - rsM2.get(c)
				).reduce((double) 1, (currentVal, val) -> currentVal * val))
			.toArray());
		return rsOut;
	}

	// NOTE: this is the best estimation possible when we only have the two row sparsity vectors
	private RSVector estimInternMM(RSVector rsM1, RSVector rsM2) {
		// double avgRsM2 = DoubleStream.of(rsM2).average().orElse(0);
		// RSVector rsOut = DoubleStream.of(rsM1).map(
		// 	rsM1I -> (double) 1 - Math.pow((double) 1 - (rsM1I * avgRsM2), rsM2.length)).toArray();
		RSVector rsOut = rsM1.map(
			rsM1I -> (double) 1 - rsM2.reduce((double) 1,
				(currentVal, rsM2J) -> currentVal * ((double) 1 - (rsM1I * rsM2J))));
		return rsOut;
	}

	private RSVector estimInternCBind(RSVector rsM1, RSVector rsM2) {
		return new RSVector(IntStream.range(0, rsM1.size()).mapToDouble(
			idx -> (rsM1.get(idx) + rsM2.get(idx)) / (double) 2).toArray());
	}

	private RSVector estimInternRBind(RSVector rsM1, RSVector rsM2) {
		return rsM1.append(rsM2);
	}

	private RSVector getRowWiseSparsityVector(MatrixBlock mb) {
		int numRows = mb.getNumRows();
		if(mb.isInSparseFormat()) {
			double[] rsArray = new double[numRows];
			for(int counter = 0; counter < numRows; counter++) {
				SparseRow sparseRow = mb.getSparseBlock().get(counter);
				rsArray[counter] = (sparseRow == null) ? 0 : (double) sparseRow.size() / mb.getNumColumns();
			}
			return new RSVector(rsArray);
		}
		else {
			return new RSVector(IntStream.range(0, numRows).mapToDouble(
				rIdx -> (double) mb.getDenseBlock().countNonZeros(rIdx) / mb.getNumColumns()).toArray());
		}
	}

	private int[] getNonZeroColumnIndices(MatrixBlock mb, final int rIdx) {
		int[] nonZeroCols;
		if(mb.isInSparseFormat()) {
			SparseRow sparseRow = mb.getSparseBlock().get(rIdx);
			nonZeroCols = (sparseRow == null) ? new int[0] : sparseRow.indexes();
		}
		else {
			nonZeroCols = IntStream.range(0, mb.getNumColumns())
				.filter(cIdx -> mb.get(rIdx, cIdx) != 0).toArray();
		}
		return nonZeroCols;
	}

	public static DataCharacteristics deriveOutputCharacteristics(MMNode node, double spOut) {
		if(node.isLeaf() ||
			(node.getDataCharacteristics() != null && node.getDataCharacteristics().getNonZeros() != -1)) {
			return node.getDataCharacteristics();
		}

		MMNode nodeLeft = node.getLeft();
		MMNode nodeRight = node.getRight();
		switch(node.getOp()) {
			case MM:
				return new MatrixCharacteristics(nodeLeft.getRows(), nodeRight.getCols(),
					OptimizerUtils.getNnz(nodeLeft.getRows(), nodeRight.getCols(), spOut));
			case MULT:
			case PLUS:
			case NEQZERO:
			case EQZERO:
				return new MatrixCharacteristics(nodeLeft.getRows(), nodeLeft.getCols(),
					OptimizerUtils.getNnz(nodeLeft.getRows(), nodeLeft.getCols(), spOut));
			case RBIND:
				return new MatrixCharacteristics(nodeLeft.getRows()+nodeLeft.getRows(), nodeLeft.getCols(),
					OptimizerUtils.getNnz(nodeLeft.getRows()+nodeRight.getRows(), nodeLeft.getCols(), spOut));
			case CBIND:
				return new MatrixCharacteristics(nodeLeft.getRows(), nodeLeft.getCols()+nodeRight.getCols(),
					OptimizerUtils.getNnz(nodeLeft.getRows(), nodeLeft.getCols()+nodeRight.getCols(), spOut));
			case DIAG:
				int ncol = nodeLeft.getCols()==1 ? nodeLeft.getRows() : 1;
				return new MatrixCharacteristics(nodeLeft.getRows(), ncol,
					OptimizerUtils.getNnz(nodeLeft.getRows(), ncol, spOut));
			case TRANS:
			case RESHAPE:
				throw new NotImplementedException("Characteristics derivation for trans and reshape has not been " +
					"implemented yet, but could be implemented similar to EstimatorMatrixHistogram.java");
			default:
				throw new NotImplementedException();
		}
	}

	public static class RSVector {
		private final double[] rs;

		public RSVector(double[] rs) {
			this.rs = rs;
		}

		public double[] get() {
			return this.rs;
		}

		public double get(int idx) {
			return this.rs[idx];
		}

		public int size() {
			return this.rs.length;
		}

		public double avg() {
			return DoubleStream.of(this.rs).average().orElse(0);
		}

		public RSVector append(RSVector that) {
			return new RSVector(ArrayUtils.addAll(this.rs, that.get()));
		}

		public RSVector map(DoubleUnaryOperator mapper) {
			return new RSVector(DoubleStream.of(this.rs).map(mapper).toArray());
		}

		public double reduce(double identity, DoubleBinaryOperator op) {
			return DoubleStream.of(this.rs).reduce(identity, op);
		}
	};
};
