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

	private void estimInternChain(MMNode node) {
		estimInternChain(node, null, null);
	}

	private void estimInternChain(MMNode node, double[] rsRightNeighbor, OpCode opRightNeighbor) {
		double[] rsOut;
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
					estimInternChain(node.getLeft(), (double[])(node.getRight().getSynopsis()), node.getOp());
					rsOut = (double[])node.getLeft().getSynopsis();
					break;
				case CBIND:
					/** NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into a cbind operation when having only row sparsity vectors
					 */
					estimInternChain(node.getLeft());
					estimInternChain(node.getRight());
					double[] rsCBind = estimInternCBind((double[])(node.getLeft().getSynopsis()), (double[])(node.getRight().getSynopsis()));
					if(rsRightNeighbor != null) {
						rsOut = (double[])estimInternMMFallback(rsCBind, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = (double[])rsCBind;
					break;
				case RBIND:
					/** NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an rbind operation when having only row sparsity vectors
					 */
					estimInternChain(node.getLeft());
					estimInternChain(node.getRight());
					double[] rsRBind = estimInternRBind((double[])(node.getLeft().getSynopsis()), (double[])(node.getRight().getSynopsis()));
					if(rsRightNeighbor != null) {
						rsOut = (double[])estimInternMMFallback(rsRBind, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = (double[])rsRBind;
					break;
				case PLUS:
					/** NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an element-wise operation when having only row sparsity vectors
					 */
					estimInternChain(node.getLeft());
					estimInternChain(node.getRight());
					double[] rsPlus = estimInternPlus((double[])(node.getLeft().getSynopsis()), (double[])(node.getRight().getSynopsis()));
					if(rsRightNeighbor != null) {
						rsOut = (double[])estimInternMMFallback(rsPlus, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = (double[])rsPlus;
					break;
				case MULT:
					/** NOTE: considering the current node as new DAG for estimation (cut), since the row sparsity of
					 * the right neighbor cannot be aggregated into an element-wise operation when having only row sparsity vectors
					 */
					estimInternChain(node.getLeft());
					estimInternChain(node.getRight());
					double[] rsMult = estimInternMult((double[])(node.getLeft().getSynopsis()), (double[])(node.getRight().getSynopsis()));
					if(rsRightNeighbor != null) {
						rsOut = (double[])estimInternMMFallback(rsMult, rsRightNeighbor);
						if(opRightNeighbor != OpCode.MM)
							throw new NotImplementedException("Fallback sparsity estimation has only been " +
								"considered for MM operation w/ right neighbor yet.");
					}
					else
						rsOut = (double[])rsMult;
					break;
				default:
					throw new NotImplementedException("Chain estimation for operator " + node.getOp().toString() +
					" is not supported yet.");
			}
		}
		node.setSynopsis(rsOut);
		node.setDataCharacteristics(deriveOutputCharacteristics(node, DoubleStream.of(rsOut).average().orElse(0)));
		return;
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

	// Corresponds to Algorithm 1 in the publication
	private double[] estimInternMM(MatrixBlock m1, double[] rsM2) {
		double[] rsOut = IntStream.range(0, m1.getNumRows()).mapToDouble(
			r -> (double) 1 - IntStream.of(getNonZeroColumnIndices(m1, r)).mapToDouble(
					c -> (double) 1 - rsM2[c]
				).reduce((double) 1, (currentVal, val) -> currentVal * val))
			.toArray();
		return rsOut;
	}

	// NOTE: this is the best estimation possible when we only have the two row sparsity vectors
	private double[] estimInternMMFallback(double[] rsM1, double[] rsM2) {
		// NOTE: Considering the average would probably not be far off while saving computing time
		// double avgRsM2 = DoubleStream.of(rsM2).average().orElse(0);
		// double[] rsOut = DoubleStream.of(rsM1).map(
		// 	rsM1I -> (double) 1 - Math.pow((double) 1 - (rsM1I * avgRsM2), rsM2.length)).toArray();
		double[] rsOut = DoubleStream.of(rsM1).map(
			rsM1I -> (double) 1 - DoubleStream.of(rsM2).reduce((double) 1,
				(currentVal, rsM2J) -> currentVal * ((double) 1 - (rsM1I * rsM2J)))).toArray();
		return rsOut;
	}

	private double[] estimInternCBind(double[] rsM1, double[] rsM2) {
		// FIXME: this assumes that the number of columns is equivalent for both inputs
		return IntStream.range(0, rsM1.length).mapToDouble(
			idx -> (rsM1[idx] + rsM2[idx]) / (double) 2).toArray();
	}

	private double[] estimInternRBind(double[] rsM1, double[] rsM2) {
		return ArrayUtils.addAll(rsM1, rsM2);
	}

	private double[] estimInternPlus(double[] rsM1, double[] rsM2) {
		// row-wise average case estimates
		// rsM1 + rsM2 - (rsM1 * rsM2)
		return IntStream.range(0, rsM1.length).mapToDouble(
			idx -> rsM1[idx] + rsM2[idx] - (rsM1[idx] * rsM2[idx])).toArray();
	}

	private double[] estimInternMult(double[] rsM1, double[] rsM2) {
		// row-wise average case estimates
		// rsM1 * rsM2
		return IntStream.range(0, rsM1.length).mapToDouble(
			idx -> rsM1[idx] * rsM2[idx]).toArray();
	}

	private double[] estimInternDiag(MatrixBlock mb) {
		double[] rsOut = IntStream.range(0, mb.getNumRows()).mapToDouble(
				rIdx -> (mb.get(rIdx, rIdx) == 0) ? 0d : 1d)
			.toArray();
		return rsOut;
	}

	private double[] getRowWiseSparsityVector(MatrixBlock mb) {
		int numRows = mb.getNumRows();
		if(mb.isInSparseFormat()) {
			double[] rsArray = new double[numRows];
			for(int counter = 0; counter < numRows; counter++) {
				SparseRow sparseRow = mb.getSparseBlock().get(counter);
				rsArray[counter] = (sparseRow == null) ? 0 : (double) sparseRow.size() / mb.getNumColumns();
			}
			return rsArray;
		}
		else {
			return IntStream.range(0, numRows).mapToDouble(
					rIdx -> (double) mb.getDenseBlock().countNonZeros(rIdx) / mb.getNumColumns())
				.toArray();
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
