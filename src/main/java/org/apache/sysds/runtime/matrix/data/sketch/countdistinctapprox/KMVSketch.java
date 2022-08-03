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

package org.apache.sysds.runtime.matrix.data.sketch.countdistinctapprox;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperatorTypes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.utils.Hash;

/**
 * KMV synopsis(for k minimum values) Distinct-Value Estimation
 *
 * Kevin S. Beyer, Peter J. Haas, Berthold Reinwald, Yannis Sismanis, Rainer Gemulla:
 *
 * On synopses for distinct‐value estimation under multiset operations. SIGMOD 2007
 *
 * TODO: Add multi-threaded version
 *
 */
public class KMVSketch extends CountDistinctApproxSketch {

	private static final Log LOG = LogFactory.getLog(KMVSketch.class.getName());

	public KMVSketch(Operator op) {
		super(op);
	}

	@Override
	public MatrixBlock getValue(MatrixBlock blkIn) {

		if (this.op.getDirection().isRowCol()) {
			// D is the number of possible distinct values in the MatrixBlock.
			// plus 1 to take account of 0 input.
			long D = blkIn.getNonZeros() + 1;

			/**
			 * To ensure that the likelihood to hash to the same value we need O(D^2) positions to hash to assign. If the
			 * value is higher than int (which is the area we hash to) then use Integer Max value as largest hashing space.
			 */
			long tmp = D * D;
			int M = (tmp > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) tmp;
			/**
			 * The estimator is asymptotically unbiased as k becomes large, but memory usage also scales with k. Furthermore k
			 * value must be within range: D >> k >> 0
			 */
			int k = D > 64 ? 64 : (int) D;

			SmallestPriorityQueue spq = getKSmallestHashes(blkIn, k, M);

			if(LOG.isDebugEnabled()) {
				LOG.debug("M not forced to int size: " + tmp);
				LOG.debug("M: " + M);
				LOG.debug("M: " + M);
				LOG.debug("kth smallest hash:" + spq.peek());
				LOG.debug("spq: " + spq);
			}


			long res = countDistinctValuesKMV(spq, k, M, D);
			if(res <= 0) {
				throw new DMLRuntimeException("Impossible estimate of distinct values");
			}

			// Result is a 1x1 matrix block
			return new MatrixBlock(res);

		} else if (this.op.getDirection().isRow()) {
			long D = (long) Math.floor(blkIn.getNonZeros() / (double) blkIn.getNumRows()) + 1;
			long tmp = D * D;
			int M = (tmp > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) tmp;
			int k = D > 64 ? 64 : (int) D;

			MatrixBlock resultMatrix = new MatrixBlock(blkIn.getNumRows(), 1, false, blkIn.getNumRows());
			resultMatrix.allocateBlock();

			SmallestPriorityQueue spq = new SmallestPriorityQueue(k);
			for (int i=0; i<blkIn.getNumRows(); ++i) {
				for (int j=0; j<blkIn.getNumColumns(); ++j) {
					spq.add(blkIn.getValue(i, j));
				}

				long res = countDistinctValuesKMV(spq, k, M, D);
				resultMatrix.setValue(i, 0, res);

				spq.clear();
			}

			return resultMatrix;

		} else {  // Col
			long D = (long) Math.floor(blkIn.getNonZeros() / (double) blkIn.getNumColumns()) + 1;
			long tmp = D * D;
			int M = (tmp > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) tmp;
			int k = D > 64 ? 64 : (int) D;

			MatrixBlock resultMatrix = new MatrixBlock(1, blkIn.getNumColumns(), false, blkIn.getNumColumns());
			resultMatrix.allocateBlock();

			SmallestPriorityQueue spq = new SmallestPriorityQueue(k);
			for (int j=0; j<blkIn.getNumColumns(); ++j) {
				for (int i=0; i<blkIn.getNumRows(); ++i) {
					spq.add(blkIn.getValue(i, j));
				}

				long res = countDistinctValuesKMV(spq, k, M, D);
				resultMatrix.setValue(0, j, res);

				spq.clear();
			}

			return resultMatrix;
		}
	}

	private SmallestPriorityQueue getKSmallestHashes(MatrixBlock in, int k, int M) {
		SmallestPriorityQueue spq = new SmallestPriorityQueue(k);
		countDistinctValuesKMV(in, op.getHashType(), k, spq, M);

		return spq;
	}

	private void countDistinctValuesKMV(MatrixBlock in, Hash.HashType hashType, int k, SmallestPriorityQueue spq,
		int m) {
		double[] data;
		if(in.isEmpty())
			spq.add(0);
		else if(in instanceof CompressedMatrixBlock)
			throw new NotImplementedException("Cannot approximate distinct count for compressed matrices");
		else if(in.getSparseBlock() != null) {
			SparseBlock sb = in.getSparseBlock();
			if(sb.isContiguous()) {
				data = sb.values(0);
				countDistinctValuesKMV(data, hashType, k, spq, m);
			}
			else {
				for(int i = 0; i < in.getNumRows(); i++) {
					if(!sb.isEmpty(i)) {
						data = sb.values(i);
						countDistinctValuesKMV(data, hashType, k, spq, m);
					}
				}
			}
		}
		else {
			DenseBlock db = in.getDenseBlock();
			final int bil = db.index(0);
			final int biu = db.index(in.getNumRows());
			for(int i = bil; i <= biu; i++) {
				data = db.valuesAt(i);
				countDistinctValuesKMV(data, hashType, k, spq, m);
			}
		}
	}

	private void countDistinctValuesKMV(double[] data, Hash.HashType hashType, int k, SmallestPriorityQueue spq, int m) {
		for(double fullValue : data) {
			int hash = Hash.hash(fullValue, hashType);
			int v = (Math.abs(hash)) % (m - 1) + 1;
			spq.add(v);
		}
	}

	private long countDistinctValuesKMV(SmallestPriorityQueue spq, int k, int M, long D) {
		long res;
		if(spq.size() < k) {
			res = spq.size();
		}
		else {
			double kthSmallestHash = spq.poll();
			double U_k = kthSmallestHash / (double) M;
			double estimate = (double) (k - 1) / U_k;
			double ceilEstimate = Math.min(estimate, (double) D);

			if(LOG.isDebugEnabled()) {
				LOG.debug("U_k : " + U_k);
				LOG.debug("Estimate: " + estimate);
				LOG.debug("Ceil worst case: " + D);
			}
			res = Math.round(ceilEstimate);
		}

		return res;
	}

	@Override
	public MatrixBlock getValueFromSketch(CorrMatrixBlock arg0) {
		MatrixBlock blkIn = arg0.getValue();
		if(op.getDirection().isRow()) {
			// 1000 x 1 blkOut
			MatrixBlock blkOut = new MatrixBlock(blkIn.getNumRows(), 1, false, blkIn.getNumRows());
			blkOut.allocateBlock();
			for(int i = 0; i < blkIn.getNumRows(); ++i) {
				getDistinctCountFromSketchByIndex(arg0, i, blkOut);
			}

			return blkOut;
		}
		else if(op.getDirection().isCol()) {
			// 1 x 1000 blkOut
			MatrixBlock blkOut = new MatrixBlock(1, blkIn.getNumColumns(), false, blkIn.getNumColumns());
			blkOut.allocateBlock();
			for(int j = 0; j < blkIn.getNumColumns(); ++j) {
				getDistinctCountFromSketchByIndex(arg0, j, blkOut);
			}

			return blkOut;
		}
		else { // op.getDirection().isRowCol()

			// 1 x 1 blkOut
			MatrixBlock blkOut = new MatrixBlock(1, 1, false, 1);
			blkOut.allocateBlock();
			getDistinctCountFromSketchByIndex(arg0, 0, blkOut);

			return blkOut;
		}
	}

	private void getDistinctCountFromSketchByIndex(CorrMatrixBlock arg0, int idx, MatrixBlock blkOut) {
		MatrixBlock blkIn = arg0.getValue();
		MatrixBlock blkInCorr = arg0.getCorrection();

		if(op.getOperatorType() != CountDistinctOperatorTypes.KMV) {
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot use " + op.getOperatorType());
		}

		double kthSmallestHash;
		if(op.getDirection().isRow() || op.getDirection().isRowCol()) {
			kthSmallestHash = blkIn.getValue(idx, 0);
		}
		else { // op.getDirection().isCol()
			kthSmallestHash = blkIn.getValue(0, idx);
		}

		double nHashes = blkInCorr.getValue(idx, 0);
		double k = blkInCorr.getValue(idx, 1);
		double D = blkInCorr.getValue(idx, 2);

		double D2 = D * D;
		double M = (D2 > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : D2;

		double ceilEstimate;
		if(nHashes != 0 && nHashes < k) {
			ceilEstimate = nHashes;
		}
		else if(nHashes == 0) {
			ceilEstimate = 1;
		}
		else {
			double U_k = kthSmallestHash / M;
			double estimate = (k - 1) / U_k;
			ceilEstimate = Math.min(estimate, D);
		}

		if(op.getDirection().isRow() || op.getDirection().isRowCol()) {
			blkOut.setValue(idx, 0, ceilEstimate);
		}
		else { // op.getDirection().isCol()
			blkOut.setValue(0, idx, ceilEstimate);
		}
	}

	// Create sketch
	@Override
	public CorrMatrixBlock create(MatrixBlock blkIn) {

		// We need a matrix containing sketch metadata per block
		// N x 3 row vector: (nHashes, k, D)
		// O(N) extra space

		if(op.getDirection().isRowCol()) {
			// (nHashes, k, D) row matrix
			MatrixBlock blkOut = new MatrixBlock(blkIn);
			MatrixBlock blkOutCorr = new MatrixBlock(1, 3, false);
			createSketchByIndex(blkIn, blkOutCorr, 0, blkOut);
			return new CorrMatrixBlock(blkOut, blkOutCorr);
		}
		else if(op.getDirection().isRow()) {
			MatrixBlock blkOut = blkIn;
			MatrixBlock blkOutCorr = new MatrixBlock(blkIn.getNumRows(), 3, false);
			// (nHashes, k, D) row matrix
			for(int i = 0; i < blkIn.getNumRows(); ++i) {
				createSketchByIndex(blkOut, blkOutCorr, i);
			}
			return new CorrMatrixBlock(blkOut, blkOutCorr);

		}
		else if(op.getDirection().isCol()) {
			MatrixBlock blkOut = blkIn;
			// (nHashes, k, D) row matrix
			MatrixBlock blkOutCorr = new MatrixBlock(blkIn.getNumColumns(), 3, false);
			for(int j = 0; j < blkIn.getNumColumns(); ++j) {
				createSketchByIndex(blkOut, blkOutCorr, j);
			}
			return new CorrMatrixBlock(blkOut, blkOutCorr);
		}
		else {
			throw new DMLRuntimeException(String.format("Unexpected direction: %s", op.getDirection()));
		}
	}

	private MatrixBlock sliceMatrixBlockByIndexDirection(MatrixBlock blkIn, int idx) {
		MatrixBlock blkInSlice;
		if(op.getDirection().isRow()) {
			blkInSlice = blkIn.slice(idx, idx);
		}
		else if(op.getDirection().isCol()) {
			blkInSlice = blkIn.slice(0, blkIn.getNumRows() - 1, idx, idx);
		}
		else {
			blkInSlice = blkIn;
		}

		return blkInSlice;
	}

	private void createSketchByIndex(MatrixBlock blkIn, MatrixBlock sketchMetaMB, int idx) {
		createSketchByIndex(blkIn, sketchMetaMB, idx, null);
	}

	private void createSketchByIndex(MatrixBlock blkIn, MatrixBlock sketchMetaMB, int idx, MatrixBlock blkOut) {

		MatrixBlock sketchMB = (blkOut == null) ? blkIn : blkOut;

		MatrixBlock blkInSlice = sliceMatrixBlockByIndexDirection(blkIn, idx);
		long D = blkInSlice.getNonZeros() + 1;

		long D2 = D * D;
		int M = (D2 > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) D2;
		int k = D > 64 ? 64 : (int) D;

		// blkOut is only passed as parameter in case dir == RowCol
		// This means that the entire block will produce a single 1xK sketch-
		// The output matrix block must be resized and filled with 0 accordingly
		if(blkOut != null) {
			sketchMB.reset(1, k);
		}

		if(blkInSlice.getLength() == 1 || blkInSlice.isEmpty()) {

			// There can only be 1 distinct value for a 1x1 or empty matrix
			// getMatrixValue() will short circuit and return 1 if nHashes = 0

			// (nHashes, k, D) row matrix
			sketchMetaMB.setValue(idx, 0, 0);
			sketchMetaMB.setValue(idx, 1, k);
			sketchMetaMB.setValue(idx, 2, D);

			return;
		}

		SmallestPriorityQueue spq = getKSmallestHashes(blkInSlice, k, M);
		int nHashes = spq.size();
		assert (nHashes > 0);

		// nHashes != k always

		int i = 0;
		while(!spq.isEmpty()) {
			double toInsert = spq.poll();
			if(op.getDirection().isRow()) {
				sketchMB.setValue(idx, i, toInsert);
			}
			else if(op.getDirection().isCol()) {
				sketchMB.setValue(i, idx, toInsert);
			}
			else {
				sketchMB.setValue(idx, i, toInsert);
			}
			++i;
		}

		// Last column contains the correction
		sketchMetaMB.setValue(idx, 0, nHashes);
		sketchMetaMB.setValue(idx, 1, k);
		sketchMetaMB.setValue(idx, 2, D);
	}

	@Override
	public CorrMatrixBlock union(CorrMatrixBlock arg0, CorrMatrixBlock arg1) {

		// Both matrices are guaranteed to be row-/column-aligned
		MatrixBlock matrix0 = arg0.getValue();
		MatrixBlock matrix1 = arg1.getValue();

		if(op.getDirection().isRow()) {
			// Use the wider of the 2 inputs for stable aggregation.
			// The number of rows is always guaranteed to match due to col index function execution.
			// Therefore, checking the number of columns is sufficient.
			MatrixBlock combined;
			if(matrix0.getNumColumns() > matrix1.getNumColumns()) {
				combined = matrix0;
			}
			else {
				combined = matrix1;
			}
			// (nHashes, k, D)
			MatrixBlock combinedCorr = new MatrixBlock(matrix0.getNumRows(), 3, false);

			CorrMatrixBlock blkout = new CorrMatrixBlock(combined, combinedCorr);
			for(int i = 0; i < matrix0.getNumRows(); ++i) {
				unionSketchByIndex(arg0, arg1, i, blkout);
			}

			return blkout;

		}
		else if(op.getDirection().isCol()) {
			// Use the taller of the 2 inputs for stable aggregation.
			// The number of columns is always guaranteed to match due to col index function execution.
			// Therefore, checking the number of rows is sufficient.
			MatrixBlock combined;
			if(matrix0.getNumRows() > matrix1.getNumRows()) {
				combined = matrix0;
			}
			else {
				combined = matrix1;
			}
			// (nHashes, k, D) row vector
			MatrixBlock combinedCorr = new MatrixBlock(matrix0.getNumColumns(), 3, false);

			CorrMatrixBlock blkOut = new CorrMatrixBlock(combined, combinedCorr);
			for(int j = 0; j < matrix0.getNumColumns(); ++j) {
				unionSketchByIndex(arg0, arg1, j, blkOut);
			}

			return blkOut;

		}
		else { // op.getDirection().isRowCol()

			// Use the wider of the 2 inputs for stable aggregation.
			// The number of rows is always guaranteed to match due to col index function execution.
			// Therefore, checking the number of columns is sufficient.
			MatrixBlock combined;
			if(matrix0.getNumColumns() > matrix1.getNumColumns()) {
				combined = matrix0;
			}
			else {
				combined = matrix1;
			}
			// (nHashes, k, D)
			MatrixBlock combinedCorr = new MatrixBlock(1, 3, false);

			CorrMatrixBlock blkOut = new CorrMatrixBlock(combined, combinedCorr);
			unionSketchByIndex(arg0, arg1, 0, blkOut);

			return blkOut;
		}
	}

	public void unionSketchByIndex(CorrMatrixBlock arg0, CorrMatrixBlock arg1, int idx, CorrMatrixBlock blkOut) {
		MatrixBlock corr0 = arg0.getCorrection();
		MatrixBlock corr1 = arg1.getCorrection();

		validateSketchMetadata(corr0);
		validateSketchMetadata(corr1);

		// Both matrices are guaranteed to be row-/column-aligned
		MatrixBlock matrix0 = arg0.getValue();
		MatrixBlock matrix1 = arg1.getValue();

		if((op.getDirection().isRow() && matrix0.getNumRows() != matrix1.getNumRows()) ||
			(op.getDirection().isCol() && matrix0.getNumColumns() != matrix1.getNumColumns())) {
			throw new DMLRuntimeException("Cannot take the union of sketches: rows/columns are not aligned");
		}

		MatrixBlock combined = blkOut.getValue();
		MatrixBlock combinedCorr = blkOut.getCorrection();

		double nHashes0 = corr0.getValue(idx, 0);
		double k0 = corr0.getValue(idx, 1);
		double D0 = corr0.getValue(idx, 2);

		double nHashes1 = corr1.getValue(idx, 0);
		double k1 = corr1.getValue(idx, 1);
		double D1 = corr1.getValue(idx, 2);

		double nHashes = Math.max(nHashes0, nHashes1);
		double k = Math.max(k0, k1);
		double D = D0 + D1 - 1;

		SmallestPriorityQueue hashUnion = new SmallestPriorityQueue((int) nHashes);

		for(int i = 0; i < nHashes0; ++i) {
			double val;
			if(op.getDirection().isRow() || op.getDirection().isRowCol()) {
				val = matrix0.getValue(idx, i);
			}
			else { // op.getDirection().isCol()
				val = matrix0.getValue(i, idx);
			}
			hashUnion.add(val);
		}

		for(int i = 0; i < nHashes1; ++i) {
			double val;
			if(op.getDirection().isRow() || op.getDirection().isRowCol()) {
				val = matrix1.getValue(idx, i);
			}
			else { // op.getDirection().isCol()
				val = matrix1.getValue(i, idx);
			}
			hashUnion.add(val);
		}

		int i = 0;
		while(!hashUnion.isEmpty()) {
			double val = hashUnion.poll();
			if(op.getDirection().isRow() || op.getDirection().isRowCol()) {
				combined.setValue(idx, i, val);
			}
			else { // op.getDirection().isCol()
				combined.setValue(i, idx, val);
			}
			i++;
		}

		combinedCorr.setValue(idx, 0, nHashes);
		combinedCorr.setValue(idx, 1, k);
		combinedCorr.setValue(idx, 2, D);
	}

	@Override
	public CorrMatrixBlock intersection(CorrMatrixBlock arg0, CorrMatrixBlock arg1) {
		throw new NotImplementedException(
			String.format("%s intersection has not been implemented yet", KMVSketch.class.getSimpleName()));
	}
}
