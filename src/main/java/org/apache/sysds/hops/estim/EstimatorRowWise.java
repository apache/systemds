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
		double[] rsOut = estimInternMMChain(root);
		double sparsity = DoubleStream.of(rsOut).average().orElse(0);

		MatrixCharacteristics matrixCharacteristics = getMatrixCharacteristics(root, sparsity);

		return root.setDataCharacteristics(matrixCharacteristics);
	}

	@Override 
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getDataCharacteristics(),
				m2.getDataCharacteristics(), op).getSparsity();

		double[] rsOut = estimIntern(m1, m2, op);
		return DoubleStream.of(rsOut).average().orElse(0);
	}

	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getDataCharacteristics(), null, op).getSparsity();
		throw new NotImplementedException();
	}

	private double[] estimInternMMChain(MMNode node) {
		return estimInternMMChain(node, null, null);
	}

	private double[] estimInternMMChain(MMNode node, double[] rsRightNeighbor, OpCode opRightNeighbor) {
		if(node.isLeaf()) {
			MatrixBlock mb = node.getData();
			if(rsRightNeighbor == null)
				return getRowWiseSparsityVector(mb);
			else
				return estimIntern(mb, rsRightNeighbor, opRightNeighbor);
		}
		switch(node.getOp()) {
			case MM:
				double[] rsRightNode = estimInternMMChain(node.getRight(), rsRightNeighbor, opRightNeighbor);
				return estimInternMMChain(node.getLeft(), rsRightNode, node.getOp());
			case CBIND:
			case RBIND:
				// consider the current node as new DAG for estimation (cut)
				double[] rsOut = estimInternBind(estimInternMMChain(node.getLeft()),
					estimInternMMChain(node.getRight()), node.getOp());
				if(rsRightNeighbor != null) {
					rsOut = estimInternMM(rsOut, rsRightNeighbor);
				}
				return rsOut;
			default:
				throw new NotImplementedException();
		}
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
			case RBIND:
				return estimInternBind(getRowWiseSparsityVector(m1), rsM2, op);
			default:
				throw new NotImplementedException("Sparsity estimation for operation " + op.toString() + " not supported yet.");
		}
	}

	// Corresponds to Algorithm 1 in the publication
	private double[] estimInternMM(MatrixBlock m1, double[] rsM2) {
		double[] rsOut = new double[m1.getNumRows()];
		for(int r = 0; r < m1.getNumRows(); r++) {
			int nonZeroCols[] = getNonZeroColumnIndices(m1, r);
			double temp = 1;
			for(int c : nonZeroCols) {
				temp *= (double) 1 - rsM2[c];
			}
			rsOut[r] = (double) 1 - temp;
		}
		return rsOut;
	}

	private double[] estimInternMM(double[] rsM1, double[] rsM2) {
		double[] rsOut = DoubleStream.of(rsM1).map(
			rsM1I -> (double) 1 - DoubleStream.of(rsM2).reduce((double) 1,
				(currentVal, rsM2J) -> currentVal * ((double) 1 - (rsM1I * rsM2J)))).toArray();
		return rsOut;
	}

	private double[] estimInternBind(double[] rsM1, double[] rsM2, OpCode op) {
		switch(op) {
			case CBIND:
				return IntStream.range(0, rsM1.length)
					.mapToDouble(idx -> (double) rsM1[idx] + rsM2[idx]).toArray();
			case RBIND:
				return ArrayUtils.addAll(rsM1, rsM2);
			default:
				throw new DMLRuntimeException("We should never reach this point.");
		}
	}

	private MatrixCharacteristics getMatrixCharacteristics(MMNode root, double sparsity) {
		switch(root.getOp()) {
			case MM:
				MMNode tmpNode = root;
				while(!tmpNode.isLeaf()) {
					tmpNode = tmpNode.getLeft();
				}
				int numRows = tmpNode.getData().getNumRows();
				tmpNode = root;
				while(!tmpNode.isLeaf()) {
					tmpNode = tmpNode.getRight();
				}
				int numColumns = tmpNode.getData().getNumColumns();
				
				return new MatrixCharacteristics(
					numRows, numColumns, (long)(numRows * numColumns * sparsity));
			default:
				throw new NotImplementedException();
		}
	}

	private double[] getRowWiseSparsityVector(MatrixBlock mb) {
		int numRows = mb.getNumRows();
		double[] rs = new double[numRows];
		if(mb.isInSparseFormat()) {
			for(int counter = 0; counter < numRows; counter++) {
				SparseRow sparseRow = mb.getSparseBlock().get(counter);
				rs[counter] = (sparseRow == null) ? 0 : (double) sparseRow.size() / mb.getNumColumns();
			}
		}
		else {
			for(int counter = 0; counter < numRows; counter++) {
				rs[counter] = (double) mb.getDenseBlock().countNonZeros(counter) / mb.getNumColumns();
			}
		}
		return rs;
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
};
