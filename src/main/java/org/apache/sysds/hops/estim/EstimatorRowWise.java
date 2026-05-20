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
		double sparsity = DoubleStream.of((double[])root.getSynopsis()).average().orElse(0);

		DataCharacteristics outputCharacteristics = deriveOutputCharacteristics(root, sparsity);
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

		double[] rsOut = estimIntern(m1, m2, op);
		return DoubleStream.of(rsOut).average().orElse(0);
	}

	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		if( isExactMetadataOp(op, m1.getNumColumns()) )
			return estimExactMetaData(m1.getDataCharacteristics(), null, op).getSparsity();

		double[] rsOut = estimIntern(m1, op);
		return DoubleStream.of(rsOut).average().orElse(0);
	}

	private double[] estimInternChain(MMNode node) {
		return estimInternChain(node, null, null);
	}

	private double[] estimInternChain(MMNode node, double[] rsRightNeighbor, OpCode opRightNeighbor) {
		double[] rsOut;
		if(node.isLeaf()) {
			MatrixBlock mb = node.getData();
			if(rsRightNeighbor != null)
				rsOut = estimIntern(mb, rsRightNeighbor, opRightNeighbor);
			else
				rsOut = getRowWiseSparsityVector(mb);
		}
		else {
			MMNode nodeLeft = node.getLeft();
			MMNode nodeRight = node.getRight();
			switch(node.getOp()) {
				case MM:
					double[] rsRightMM = estimInternChain(nodeRight, rsRightNeighbor, opRightNeighbor);
					rsOut = estimInternChain(nodeLeft, rsRightMM, node.getOp());
					break;
				case CBIND:
					/**
					 * NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into a cbind operation when having only row sparsity vectors
					 */
					double[] rsLeftCBind = estimInternChain(nodeLeft);
					double[] rsRightCBind = estimInternChain(nodeRight);
					double[] rsCBind = estimInternCBind(rsLeftCBind, rsRightCBind);
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
					double[] rsLeftRBind = estimInternChain(nodeLeft);
					double[] rsRightRBind = estimInternChain(nodeRight);
					double[] rsRBind = estimInternRBind(rsLeftRBind, rsRightRBind);
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
					double[] rsLeftPlus = estimInternChain(nodeLeft);
					double[] rsRightPlus = estimInternChain(nodeRight);
					double[] rsPlus = estimInternPlus(rsLeftPlus, rsRightPlus);
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
					double[] rsLeftMult = estimInternChain(nodeLeft);
					double[] rsRightMult = estimInternChain(nodeRight);
					double[] rsMult = estimInternMult(rsLeftMult, rsRightMult);
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
		node.setSynopsis(rsOut);
		node.setDataCharacteristics(deriveOutputCharacteristics(node, DoubleStream.of(rsOut).average().orElse(0)));
		return rsOut;
	}

	private double[] estimIntern(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		double[] rsM2 = getRowWiseSparsityVector(m2);
		return estimIntern(m1, rsM2, op);
	}

	private double[] estimIntern(MatrixBlock m1, double[] rsM2, OpCode op) {
		switch(op) {
			case MM:
				return estimInternMM(m1, rsM2);
			case CBIND:
				return estimInternCBind(getRowWiseSparsityVector(m1), rsM2);
			case RBIND:
				return estimInternRBind(getRowWiseSparsityVector(m1), rsM2);
			case PLUS:
				return estimInternPlus(getRowWiseSparsityVector(m1), rsM2);
			case MULT:
				return estimInternMult(getRowWiseSparsityVector(m1), rsM2);
			default:
				throw new NotImplementedException("Sparsity estimation for operation " + op.toString() + " not supported yet.");
		}
	}

	private double[] estimIntern(MatrixBlock mb, OpCode op) {
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
	private double[] estimInternMM(MatrixBlock m1, double[] rsM2) {
		double[] rsOut = new double[m1.getNumRows()];
		for(int rIdx = 0; rIdx < m1.getNumRows(); rIdx++) {
			double currentVal = 1;
			for(int cIdx : getNonZeroColumnIndices(m1, rIdx)) {
				currentVal *= 1.0 - rsM2[cIdx];
			}
			rsOut[rIdx] = 1 - currentVal;
		}
		return rsOut;
	}

	/**
	 * NOTE: fallback estimate using the uniform estimator (aka average-case estimator, Naive Bayes estimator) for
	 * the case when we are limited to the row sparsity vectors of both inputs
	 * NOTE: Considering the average of the second matrix would probably not be far off while saving computing time
	 */
	private double[] estimInternMMFallback(double[] rsM1, double[] rsM2) {
		double[] rsOut = new double[rsM1.length];
		for(int i = 0; i < rsM1.length; i++) {
			double rsM1i = rsM1[i];
			if(rsM1i == 0) {
				rsOut[i] = 0;
			}
			else {
				double currentVal = 1;
				for(int j = 0; j < rsM2.length; j++) {
					currentVal *= 1.0 - (rsM1i * rsM2[j]);
				}
				rsOut[i] = 1.0 - currentVal;
			}
		}
		return rsOut;
	}

	private double[] estimInternCBind(double[] rsM1, double[] rsM2) {
		// FIXME: this estimate assumes that the number of columns is equivalent for both inputs
		double[] rsOut = new double[rsM1.length];
		for(int idx = 0; idx < rsM1.length; idx++) {
			rsOut[idx] = (rsM1[idx] + rsM2[idx]) / 2.0;
		}
		return rsOut;
	}

	private double[] estimInternRBind(double[] rsM1, double[] rsM2) {
		return ArrayUtils.addAll(rsM1, rsM2);
	}

	private double[] estimInternPlus(double[] rsM1, double[] rsM2) {
		// row-wise average case estimates
		// rsM1 + rsM2 - (rsM1 * rsM2)
		double[] rsOut = new double[rsM1.length];
		for(int idx = 0; idx < rsM1.length; idx++) {
			rsOut[idx] = rsM1[idx] + rsM2[idx] - (rsM1[idx] * rsM2[idx]);
		}
		return rsOut;
	}

	private double[] estimInternMult(double[] rsM1, double[] rsM2) {
		// row-wise average case estimates
		// rsM1 * rsM2
		double[] rsOut = new double[rsM1.length];
		for(int idx = 0; idx < rsM1.length; idx++) {
			rsOut[idx] = rsM1[idx] * rsM2[idx];
		}
		return rsOut;
	}

	private double[] estimInternDiag(MatrixBlock mb) {
		double[] rsOut = new double[mb.getNumRows()];
		for(int rIdx = 0; rIdx < mb.getNumRows(); rIdx++) {
			rsOut[rIdx] = (mb.get(rIdx, rIdx) == 0) ? 0 : 1;
		}
		return rsOut;
	}

	private double[] getRowWiseSparsityVector(MatrixBlock mb) {
		int numRows = mb.getNumRows();
		double[] rsOut = new double[numRows];
		if(mb.isInSparseFormat()) {
			for(int rIdx = 0; rIdx < numRows; rIdx++) {
				SparseRow sparseRow = mb.getSparseBlock().get(rIdx);
				rsOut[rIdx] = (sparseRow == null) ? 0 : (double) sparseRow.size() / mb.getNumColumns();
			}
		}
		else {
			for(int rIdx = 0; rIdx < numRows; rIdx++) {
				rsOut[rIdx] = (double) mb.getDenseBlock().countNonZeros(rIdx) / mb.getNumColumns();
			}
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

	public static DataCharacteristics deriveOutputCharacteristics(MMNode node, double spOut) {
		if(node.isLeaf() ||
			(node.getDataCharacteristics() != null && node.getDataCharacteristics().getNonZeros() != -1)) {
			return node.getDataCharacteristics();
		}

		MMNode nodeLeft = node.getLeft();
		MMNode nodeRight = node.getRight();
		int leftNRow = nodeLeft.getRows();
		int leftNCol = nodeLeft.getCols();
		int rightNRow = nodeRight.getRows();
		int rightNCol = nodeRight.getCols();
		switch(node.getOp()) {
			case MM:
				return new MatrixCharacteristics(leftNRow, rightNCol,
					OptimizerUtils.getNnz(leftNRow, rightNCol, spOut));
			case MULT:
			case PLUS:
			case NEQZERO:
			case EQZERO:
				return new MatrixCharacteristics(leftNRow, leftNCol,
					OptimizerUtils.getNnz(leftNRow, leftNCol, spOut));
			case RBIND:
				return new MatrixCharacteristics(leftNRow+rightNRow, leftNCol,
					OptimizerUtils.getNnz(leftNRow+rightNRow, leftNCol, spOut));
			case CBIND:
				return new MatrixCharacteristics(leftNRow, leftNCol+rightNCol,
					OptimizerUtils.getNnz(leftNRow, leftNCol+rightNCol, spOut));
			case DIAG:
				int ncol = (leftNCol == 1) ? leftNRow : 1;
				return new MatrixCharacteristics(leftNRow, ncol,
					OptimizerUtils.getNnz(leftNRow, ncol, spOut));
			case TRANS:
				return new MatrixCharacteristics(leftNCol, leftNRow,
					OptimizerUtils.getNnz(leftNCol, leftNRow, spOut));
			case RESHAPE:
				throw new NotImplementedException("Characteristics derivation for " + node.getOp() +" has not been " +
					"implemented yet, but could be implemented similar to EstimatorMatrixHistogram.java");
			default:
				throw new NotImplementedException();
		}
	}
};
