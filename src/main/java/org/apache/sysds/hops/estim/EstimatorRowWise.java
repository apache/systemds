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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.Arrays;
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

		DataCharacteristics outputCharacteristics = root.getSparsityInformation().deriveDataCharacteristics();
		return root.setDataCharacteristics(outputCharacteristics);
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( isExactMetadataOp(op, m1.getNumColumns()) ) {
			return estimExactMetaData(m1.getDataCharacteristics(),
				m2.getDataCharacteristics(), op).getSparsity();
		}

		RowwiseSparsity rsOut = estimIntern(m1, m2, op);
		return rsOut.getAvgSparsity();
	}

	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		if( isExactMetadataOp(op, m1.getNumColumns()) )
			return estimExactMetaData(m1.getDataCharacteristics(), null, op).getSparsity();

		RowwiseSparsity rsOut = estimIntern(m1, op);
		return rsOut.getAvgSparsity();
	}

	@Override
	public MMNode getInputMMNode(MatrixBlock mb) {
		MMNode node = new MMNode(mb);
		getCachedSparsityInformation(node);
		return node;
	}

	private RowwiseSparsity getCachedSparsityInformation(MMNode node) {
		if( node == null )
			return null;
		//ensure sparity information is properly cached and reused
		if( node.isLeaf() && node.getSparsityInformation() == null )
			node.setSparsityInformation(new RowwiseSparsity(node.getData()));
		else if( !node.isLeaf() )
			estim(node); //recursively obtain sparity information
		return (RowwiseSparsity) node.getSparsityInformation();
	}

	private RowwiseSparsity estimInternChain(MMNode node) {
		return estimInternChain(node, null, null);
	}

	private RowwiseSparsity estimInternChain(MMNode node, RowwiseSparsity rsRightNeighbor, OpCode opRightNeighbor) {
		RowwiseSparsity rsOut;
		if(node.isLeaf()) {
			MatrixBlock mb = node.getData();
			if(rsRightNeighbor != null)
				rsOut = estimIntern(mb, rsRightNeighbor, opRightNeighbor);
			else
				rsOut = new RowwiseSparsity(mb);
		}
		else {
			MMNode nodeLeft = node.getLeft();
			MMNode nodeRight = node.getRight();
			switch(node.getOp()) {
				case MM:
					RowwiseSparsity rsRightMM = estimInternChain(nodeRight, rsRightNeighbor, opRightNeighbor);
					if(rsRightMM == null)
						return null;
					rsOut = estimInternChain(nodeLeft, rsRightMM, node.getOp());
					break;
				case CBIND:
					/**
					 * NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into a cbind operation when having only row sparsity vectors
					 */
					RowwiseSparsity rsLeftCBind = estimInternChain(nodeLeft);
					RowwiseSparsity rsRightCBind = estimInternChain(nodeRight);
					if(rsRightCBind == null || rsLeftCBind == null)
						return null;
					RowwiseSparsity rsCBind = estimInternCBind(rsLeftCBind, rsRightCBind);
					if(rsRightNeighbor != null) {
						rsOut = estimInternMMFallback(rsCBind, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = rsCBind;
					break;
				case RBIND:
					/**
					 * NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an rbind operation when having only row sparsity vectors
					 */
					RowwiseSparsity rsLeftRBind = estimInternChain(nodeLeft);
					RowwiseSparsity rsRightRBind = estimInternChain(nodeRight);
					if(rsRightRBind == null || rsLeftRBind == null)
						return null;
					RowwiseSparsity rsRBind = estimInternRBind(rsLeftRBind, rsRightRBind);
					if(rsRightNeighbor != null) {
						rsOut = estimInternMMFallback(rsRBind, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = rsRBind;
					break;
				case PLUS:
					/**
					 * NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an element-wise operation when having only row sparsity vectors
					 */
					RowwiseSparsity rsLeftPlus = estimInternChain(nodeLeft);
					RowwiseSparsity rsRightPlus = estimInternChain(nodeRight);
					if(rsRightPlus == null || rsLeftPlus == null)
						return null;
					RowwiseSparsity rsPlus = estimInternPlus(rsLeftPlus, rsRightPlus);
					if(rsRightNeighbor != null) {
						rsOut = estimInternMMFallback(rsPlus, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = rsPlus;
					break;
				case MULT:
					/**
					 * NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an element-wise operation when having only row sparsity vectors
					 */
					RowwiseSparsity rsLeftMult = estimInternChain(nodeLeft);
					RowwiseSparsity rsRightMult = estimInternChain(nodeRight);
					if(rsRightMult == null || rsLeftMult == null)
						return null;
					RowwiseSparsity rsMult = estimInternMult(rsLeftMult, rsRightMult);
					if(rsRightNeighbor != null) {
						rsOut = estimInternMMFallback(rsMult, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = rsMult;
					break;
				default:
					throw new NotImplementedException("Chain estimation for operator " + node.getOp().toString() +
					" is not supported yet.");
			}
		}
		node.setSparsityInformation(rsOut);
		node.setDataCharacteristics(node.getSparsityInformation().deriveDataCharacteristics());
		return rsOut;
	}

	private RowwiseSparsity estimIntern(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		RowwiseSparsity rsM2 = new RowwiseSparsity(m2);
		return estimIntern(m1, rsM2, op);
	}

	private RowwiseSparsity estimIntern(MatrixBlock m1, RowwiseSparsity rsM2, OpCode op) {
		switch(op) {
			case MM:
				return estimInternMM(m1, rsM2);
			case CBIND:
				return estimInternCBind(new RowwiseSparsity(m1), rsM2);
			case RBIND:
				return estimInternRBind(new RowwiseSparsity(m1), rsM2);
			case PLUS:
				return estimInternPlus(new RowwiseSparsity(m1), rsM2);
			case MULT:
				return estimInternMult(new RowwiseSparsity(m1), rsM2);
			default:
				throw new NotImplementedException("Sparsity estimation for operation " + op.toString() + " not supported yet.");
		}
	}

	private RowwiseSparsity estimIntern(MatrixBlock mb, OpCode op) {
		switch(op) {
			case DIAG:
				return estimInternDiag(mb);
			default:
				throw new NotImplementedException("Sparsity estimation for operation " + op.toString() + " not supported yet.");
		}
	}

	/**
	 * Corresponds to Algorithm 1 in the publication
	 */
	private RowwiseSparsity estimInternMM(MatrixBlock m1, RowwiseSparsity rsM2) {
		RowwiseSparsity rsOut = new RowwiseSparsity(m1.getNumRows(), rsM2.getNumColumns());
		for(int rIdx = 0; rIdx < m1.getNumRows(); rIdx++) {
			double currentVal = 1;
			for(int cIdx : getNonZeroColumnIndices(m1, rIdx)) {
				currentVal *= 1.0 - rsM2.get(cIdx);
			}
			rsOut.set(rIdx, 1 - currentVal);
		}
		return rsOut;
	}

	/**
	 * NOTE: fallback estimate using the uniform estimator (aka average-case estimator, Naive Bayes estimator) for
	 * the case when we are limited to the row sparsity vectors of both inputs
	 * NOTE: Considering the average of the second matrix would probably not be far off while saving computing time
	 */
	private RowwiseSparsity estimInternMMFallback(RowwiseSparsity rsM1, RowwiseSparsity rsM2) {
		RowwiseSparsity rsOut = new RowwiseSparsity(rsM1.getNumRows(), rsM2.getNumColumns());
		for(int i = 0; i < rsOut.getNumRows(); i++) {
			double rsM1i = rsM1.get(i);
			if(rsM1i == 0) {
				rsOut.set(i,0);
			}
			else {
				double currentVal = 1;
				for(int j = 0; j < rsM2.getNumRows(); j++) {
					currentVal *= 1.0 - (rsM1i * rsM2.get(j));
				}
				rsOut.set(i, 1.0 - currentVal);
			}
		}
		return rsOut;
	}

	private RowwiseSparsity estimInternCBind(RowwiseSparsity rsM1, RowwiseSparsity rsM2) {
		RowwiseSparsity rsOut = new RowwiseSparsity(rsM1.getNumRows(), rsM1.getNumColumns() + rsM2.getNumColumns());
		for(int idx = 0; idx < rsOut.getNumRows(); idx++) {
			rsOut.set(idx, (rsM1.getNumColumns() * rsM1.get(idx))
				+ (rsM2.getNumColumns() * rsM2.get(idx)) / rsOut.getNumColumns());
		}
		return rsOut;
	}

	private RowwiseSparsity estimInternRBind(RowwiseSparsity rsM1, RowwiseSparsity rsM2) {
		RowwiseSparsity rsOut = new RowwiseSparsity(rsM1.getNumRows() + rsM2.getNumRows(), rsM1.getNumColumns());
		rsOut.setVector(ArrayUtils.addAll(rsM1.getVector(), rsM2.getVector()));
		return rsOut;
	}

	private RowwiseSparsity estimInternPlus(RowwiseSparsity rsM1, RowwiseSparsity rsM2) {
		// row-wise average case estimates
		// rsM1 + rsM2 - (rsM1 * rsM2)
		RowwiseSparsity rsOut = new RowwiseSparsity(rsM1.getNumRows(), rsM1.getNumColumns());
		for(int idx = 0; idx < rsOut.getNumRows(); idx++) {
			rsOut.set(idx, rsM1.get(idx) + rsM2.get(idx) - (rsM1.get(idx) * rsM2.get(idx)));
		}
		return rsOut;
	}

	private RowwiseSparsity estimInternMult(RowwiseSparsity rsM1, RowwiseSparsity rsM2) {
		// row-wise average case estimates
		// rsM1 * rsM2
		RowwiseSparsity rsOut = new RowwiseSparsity(rsM1.getNumRows(), rsM1.getNumColumns());
		for(int idx = 0; idx < rsOut.getNumRows(); idx++) {
			rsOut.set(idx, rsM1.get(idx) * rsM2.get(idx));
		}
		return rsOut;
	}

	private RowwiseSparsity estimInternDiag(MatrixBlock mb) {
		if(mb.getNumColumns() == 1)
			throw new DMLRuntimeException("Sparsity of creating a diagonal matrix from a " +
				"can be derived exactly instead of estimating.");
		RowwiseSparsity rsOut = new RowwiseSparsity(mb.getNumRows(), 1);
		for(int rIdx = 0; rIdx < mb.getNumRows(); rIdx++) {
			rsOut.set(rIdx, (mb.get(rIdx, rIdx) == 0) ? 0 : 1);
		}
		return rsOut;
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

	public static class RowwiseSparsity extends SparsityInformation{
		private double[] _row_spar;

		public RowwiseSparsity(int rlen, int clen) {
			super(rlen, clen);

			_row_spar = new double[rlen];
		}

		public RowwiseSparsity(MatrixBlock mb) {
			super(mb);

			if(mb.getLength() == mb.getNonZeros()) {
				// fully dense: constant row/column counts
				_row_spar = new double[mb.getNumRows()];
				Arrays.fill(_row_spar, 1);
			}
			else if(!mb.isEmpty()) {
				if(mb.isInSparseFormat()) {
					SparseBlock a = mb.getSparseBlock();
					_row_spar = IntStream.range(0, mb.getNumRows())
						.mapToDouble(idx -> (double)a.size(idx) / mb.getNumColumns()).toArray();
				}
				else {
					DenseBlock a = mb.getDenseBlock();
					_row_spar = IntStream.range(0, mb.getNumRows())
						.mapToDouble(idx -> (double)a.countNonZeros(idx) / mb.getNumColumns()).toArray();
				}
			}
			else {
				throw new DMLRuntimeException("Matrix Block should not be empty at this point.");
			}
		}

		@Override
		public long getNnz() {
			return Math.round(getAvgSparsity() * getNumRows() * getNumColumns());
		}

		@Override
		public double getAvgSparsity() {
			return DoubleStream.of(_row_spar).average().orElse(-1);
		}

		public double get(int rIdx) {
			return _row_spar[rIdx];
		}

		public void set(int rIdx, double val) {
			_row_spar[rIdx] = val;
		}

		public double[] getVector() {
			return _row_spar;
		}

		public void setVector(double[] vec) {
			_row_spar = vec;
		}
	}
};
