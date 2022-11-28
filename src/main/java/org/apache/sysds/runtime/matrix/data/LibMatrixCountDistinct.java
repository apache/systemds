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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysds.runtime.matrix.data.sketch.countdistinct.CountDistinctFunctionSketch;
import org.apache.sysds.runtime.matrix.data.sketch.countdistinctapprox.KMVSketch;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperatorTypes;
import org.apache.sysds.utils.Hash.HashType;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * This class contains various methods for counting the number of distinct values inside a MatrixBlock
 */
public interface LibMatrixCountDistinct {
	Log LOG = LogFactory.getLog(LibMatrixCountDistinct.class.getName());

	/**
	 * The minimum number NonZero of cells in the input before using approximate techniques for counting number of
	 * distinct values.
	 */
	int minimumSize = 1024;

	/**
	 * Public method to count the number of distinct values inside a matrix. Depending on which CountDistinctOperator
	 * selected it either gets the absolute number or a estimated value.
	 * 
	 * TODO: If the MatrixBlock type is CompressedMatrix, simply read the values from the ColGroups.
	 * 
	 * @param in the input matrix to count number distinct values in
	 * @param op the selected operator to use
	 * @return A matrix block containing the absolute distinct count for the entire input or along given row/col axis
	 */
	static MatrixBlock estimateDistinctValues(MatrixBlock in, CountDistinctOperator op) {
		if(op.getOperatorType() == CountDistinctOperatorTypes.KMV &&
			(op.getHashType() == HashType.ExpHash || op.getHashType() == HashType.StandardJava)) {
			throw new DMLException(
				"Invalid hashing configuration using " + op.getHashType() + " and " + op.getOperatorType());
		}
		else if(op.getOperatorType() == CountDistinctOperatorTypes.HLL) {
			throw new NotImplementedException("HyperLogLog has not been implemented yet");
		}

		// shortcut in the simplest case.
		if(in.getLength() == 1 || in.isEmpty()) {
			return new MatrixBlock(1);
		}

		long averageNnzPerRowOrCol;
		if (op.getDirection().isRowCol()) {
			averageNnzPerRowOrCol = in.getNonZeros();
		} else if (op.getDirection().isRow()) {
			// The average nnz per row is susceptible to skew. However, given that CP instructions is limited to
			// matrices of size at most 1000 x 1000, the performance impact of using naive counting over sketch per
			// row/col as determined by the average is negligible. Besides, the average is the simplest measure
			// available without calculating nnz per row/col.
			averageNnzPerRowOrCol = (long) Math.floor(in.getNonZeros() / (double) in.getNumRows());
		} else if (op.getDirection().isCol()) {
			averageNnzPerRowOrCol = (long) Math.floor(in.getNonZeros() / (double) in.getNumColumns());
		} else {
			throw new IllegalArgumentException("Unrecognized direction " + op.getDirection());
		}

		// Result is a dense 1x1 (RowCol), Mx1 (Row), or 1xN (Col) matrix
		MatrixBlock res;
		if (averageNnzPerRowOrCol < minimumSize) {
			// Resort to naive counting for small enough matrices
			res = countDistinctValuesNaive(in, op);
		} else {
			switch(op.getOperatorType()) {
				case COUNT:
					res = countDistinctValuesNaive(in, op);
					break;
				case KMV:
					res = new KMVSketch(op).getValue(in);
					break;
				default:
					throw new DMLException("Invalid estimator type for aggregation: " + LibMatrixCountDistinct.class.getSimpleName());
			}
		}

		return res;
	}

	/**
	 * Naive implementation of counting distinct values.
	 * 
	 * Benefit: precise, but uses memory, on the scale of inputs number of distinct values.
	 * 
	 * @param blkIn The input matrix to count number distinct values in
	 * @return A matrix block containing the absolute distinct count for the entire input or along given row/col axis
	 */
	private static MatrixBlock countDistinctValuesNaive(MatrixBlock blkIn, CountDistinctOperator op) {

		if (blkIn.isEmpty()) {
			return new MatrixBlock(1);
		}
		else if(blkIn instanceof CompressedMatrixBlock) {
			throw new NotImplementedException("countDistinct() does not support CompressedMatrixBlock");
		}

		Set<Double> distinct = new HashSet<>();
		MatrixBlock blkOut;
		double[] data;

		if (op.getDirection().isRowCol()) {
			blkOut = new MatrixBlock(1, 1, false);

			long distinctCount = 0;
			long nonZeros = blkIn.getNonZeros();

			// Check if input matrix contains any 0 values for RowCol case.
			// This does not apply to row/col case, where we count nnz per row or col during iteration.
			if(nonZeros != -1 && nonZeros < (long) blkIn.getNumColumns() * blkIn.getNumRows()) {
				distinct.add(0d);
			}

			if(blkIn.getSparseBlock() != null) {
				SparseBlock sb = blkIn.getSparseBlock();
				if(blkIn.getSparseBlock().isContiguous()) {
					// COO, CSR
					data = sb.values(0);
					distinctCount = countDistinctValuesNaive(data, distinct);
				} else {
					// MCSR
					for(int i = 0; i < blkIn.getNumRows(); i++) {
						if(!sb.isEmpty(i)) {
							data = blkIn.getSparseBlock().values(i);
							distinctCount = countDistinctValuesNaive(data, distinct);
						}
					}
				}
			} else if(blkIn.getDenseBlock() != null) {
				DenseBlock db = blkIn.getDenseBlock();
				for (int i = 0; i <= db.numBlocks(); i++) {
					data = db.valuesAt(i);
					distinctCount = countDistinctValuesNaive(data, distinct);
				}
			}

			blkOut.setValue(0, 0, distinctCount);
		} else if (op.getDirection().isRow()) {
			blkOut = new MatrixBlock(blkIn.getNumRows(), 1, false, blkIn.getNumRows());
			blkOut.allocateBlock();

			if (blkIn.getDenseBlock() != null) {
				// The naive approach would be to iterate through every (i, j) in the input. However, can do better
				// by exploiting the physical layout of dense blocks - contiguous blocks in row-major order - in memory.
				DenseBlock db = blkIn.getDenseBlock();
				for (int bix=0; bix<db.numBlocks(); ++bix) {
					data = db.valuesAt(bix);
					for (int rix=bix * db.blockSize(); rix<blkIn.getNumRows(); rix++) {
						distinct.clear();
						for (int cix=0; cix<blkIn.getNumColumns(); ++cix) {
							distinct.add(data[db.pos(rix, cix)]);
						}
						blkOut.setValue(rix, 0, distinct.size());
					}
				}
			} else if (blkIn.getSparseBlock() != null) {
				// Each sparse block type - COO, CSR, MCSR - has a different data representation, which we will exploit
				// separately.
				SparseBlock sb = blkIn.getSparseBlock();
				if (SparseBlockFactory.isSparseBlockType(sb, SparseBlock.Type.MCSR)) {
					// Currently, SparseBlockIterator only provides an interface for cell-wise iteration.
					// TODO Explore row-wise and column-wise methods for SparseBlockIterator

					// MCSR enables O(1) access to column values per row
					for (int rix=0; rix<blkIn.getNumRows(); ++rix) {
						if (sb.isEmpty(rix)) {
							continue;
						}
						distinct.clear();
						data = sb.values(rix);
						countDistinctValuesNaive(data, distinct);
						blkOut.setValue(rix, 0, distinct.size());
					}
				} else if (SparseBlockFactory.isSparseBlockType(sb, SparseBlock.Type.CSR)) {
					// Casting is safe given if-condition
					SparseBlockCSR csrBlock = (SparseBlockCSR) sb;

					// Data lies in one contiguous block in CSR format. We will iterate in row-major using O(1) op
					// size(row) to determine the number of columns per row.
					data = csrBlock.values();
					// We want to iterate through all rows to keep track of the row index for constructing the output
					for (int rix=0; rix<blkIn.getNumRows(); ++rix) {
						if (csrBlock.isEmpty(rix)) {
							continue;
						}
						distinct.clear();
						int rpos = csrBlock.pos(rix);
						int clen = csrBlock.size(rix);
						for (int colOffset=0; colOffset<clen; ++colOffset) {
							distinct.add(data[rpos + colOffset]);
						}
						blkOut.setValue(rix, 0, distinct.size());
					}
				} else { // COO
					if (!(sb instanceof SparseBlockCOO)) {
						throw new IllegalArgumentException("Input matrix is of unrecognized type: "
								+ sb.getClass().getSimpleName());
					}
					SparseBlockCOO cooBlock = (SparseBlockCOO) sb;

					// For COO, we want to avoid using pos(row) and size(row) as they use binary search, which is a
					// O(log N) op. Also, isEmpty(row) uses pos(row) internally.
					int[] rixs = cooBlock.rowIndexes();
					data = cooBlock.values();
					int i = 0;  // data iterator
					int rix = 0;  // row index
					while (rix < cooBlock.numRows() && i < rixs.length) {
						distinct.clear();
						while (i + 1 < rixs.length && rixs[i] == rixs[i + 1]) {
							distinct.add(data[i]);
							i++;
						}
						if (i + 1 < rixs.length) {  // rixs[i] != rixs[i + 1]
							distinct.add(data[i]);
						}
						blkOut.setValue(rix, 0, distinct.size());
						rix = (i + 1 < rixs.length)? rixs[i + 1] : rix;
						i++;
					}
				}
			}
		} else {  // Col aggregation
			blkOut = new MatrixBlock(1, blkIn.getNumColumns(), false, blkIn.getNumColumns());
			blkOut.allocateBlock();

			// All dense and sparse formats (COO, CSR, MCSR) are row-major formats, so there is no obvious way to iterate
			// in column-major order besides iterating through every (i, j) pair. getValue() skips over empty cells in CSR
			// and MCSR formats, but not so in COO format. This results in O(log2 R * log2 C) time for every lookup,
			// amounting to O(RC * log2R * log2C) for the whole block (R, C <= 1000 in CP case). We will eschew this
			// approach in favor of one using a hash map M of (column index, distinct values) to obtain a pseudo column-major
			// grouping of distinct values instead. Given this setup, we will simply iterate over the input
			// (according to specific dense/sparse format) in row-major order and populate M. Finally, an O(C) iteration
			// over M will yield the final result.
			Map<Integer, Set<Double>> distinctValuesByCol = new HashMap<>();
			if (blkIn.getDenseBlock() != null) {
				DenseBlock db = blkIn.getDenseBlock();
				for (int bix=0; bix<db.numBlocks(); ++bix) {
					data = db.valuesAt(bix);
					for (int cix=0; cix<blkIn.getNumColumns(); ++cix) {
						Set<Double> distinctValues = distinctValuesByCol.getOrDefault(cix, new HashSet<>());
						for (int rix=bix * db.blockSize(); rix<blkIn.getNumRows(); rix++) {
							double val = data[db.pos(rix, cix)];
							distinctValues.add(val);
						}
						distinctValuesByCol.put(cix, distinctValues);
					}
				}
			} else if (blkIn.getSparseBlock() != null) {
				SparseBlock sb = blkIn.getSparseBlock();
				if (SparseBlockFactory.isSparseBlockType(sb, SparseBlock.Type.MCSR)) {
					for (int rix=0; rix<blkIn.getNumRows(); ++rix) {
						if (sb.isEmpty(rix)) {
							continue;
						}
						int[] cixs = sb.indexes(rix);
						data = sb.values(rix);
						for (int j=0; j<sb.size(rix); ++j) {
							int cix = cixs[j];
							Set<Double> distinctValues = distinctValuesByCol.getOrDefault(cix, new HashSet<>());
							distinctValues.add(data[j]);
							distinctValuesByCol.put(cix, distinctValues);
						}
					}
				} else if (SparseBlockFactory.isSparseBlockType(sb, SparseBlock.Type.CSR)) {
					SparseBlockCSR csrBlock = (SparseBlockCSR) sb;
					data = csrBlock.values();
					for (int rix=0; rix<blkIn.getNumRows(); ++rix) {
						if (csrBlock.isEmpty(rix)) {
							continue;
						}
						int rpos = csrBlock.pos(rix);
						int clen = csrBlock.size(rix);
						int[] cixs = csrBlock.indexes();
						for (int colOffset=0; colOffset<clen; ++colOffset) {
							int cix = cixs[rpos + colOffset];
							Set<Double> distinctValues = distinctValuesByCol.getOrDefault(cix, new HashSet<>());
							distinctValues.add(data[rpos + colOffset]);
							distinctValuesByCol.put(cix, distinctValues);
						}
					}
				} else {  // COO
					if (!(sb instanceof SparseBlockCOO)) {
						throw new IllegalArgumentException("Input matrix is of unrecognized type: "
								+ sb.getClass().getSimpleName());
					}
					SparseBlockCOO cooBlock = (SparseBlockCOO) sb;

					int[] rixs = cooBlock.rowIndexes();
					int[] cixs = cooBlock.indexes();
					data = cooBlock.values();
					int i = 0;  // data iterator
					while (i < rixs.length) {
						while (i + 1 < rixs.length && rixs[i] == rixs[i + 1]) {
							int cix = cixs[i];
							Set<Double> distinctValues = distinctValuesByCol.getOrDefault(cix, new HashSet<>());
							distinctValues.add(data[i]);
							distinctValuesByCol.put(cix, distinctValues);
							i++;
						}
						if (i + 1 < rixs.length) {
							int cix = cixs[i];
							Set<Double> distinctValues = distinctValuesByCol.getOrDefault(cix, new HashSet<>());
							distinctValues.add(data[i]);
							distinctValuesByCol.put(cix, distinctValues);
						}
						i++;
					}
				}
			}
			// Fill in output block with column aggregation results
			for (int cix : distinctValuesByCol.keySet()) {
				blkOut.setValue(0, cix, distinctValuesByCol.get(cix).size());
			}
		}

		return blkOut;
	}

	private static long countDistinctValuesNaive(double[] valuesPart, Set<Double> distinct) {
		for(double v : valuesPart)
			distinct.add(v);

		return distinct.size();
	}

	static MatrixBlock countDistinctValuesFromSketch(CorrMatrixBlock arg0, CountDistinctOperator op) {
		if(op.getOperatorType() == CountDistinctOperatorTypes.COUNT)
			return new CountDistinctFunctionSketch(op).getValueFromSketch(arg0);
		else if(op.getOperatorType() == CountDistinctOperatorTypes.KMV)
			return new KMVSketch(op).getValueFromSketch(arg0);
		else if(op.getOperatorType() == CountDistinctOperatorTypes.HLL)
			throw new NotImplementedException("Not implemented yet");
		else
			throw new NotImplementedException("Not implemented yet");
	}

	static CorrMatrixBlock createSketch(MatrixBlock blkIn, CountDistinctOperator op) {
		if(op.getOperatorType() == CountDistinctOperatorTypes.COUNT)
			return new CountDistinctFunctionSketch(op).create(blkIn);
		else if(op.getOperatorType() == CountDistinctOperatorTypes.KMV)
			return new KMVSketch(op).create(blkIn);
		else if(op.getOperatorType() == CountDistinctOperatorTypes.HLL)
			throw new NotImplementedException("Not implemented yet");
		else
			throw new NotImplementedException("Not implemented yet");
	}

	static CorrMatrixBlock unionSketch(CorrMatrixBlock arg0, CorrMatrixBlock arg1, CountDistinctOperator op) {
		if(op.getOperatorType() == CountDistinctOperatorTypes.COUNT)
			return new CountDistinctFunctionSketch(op).union(arg0, arg1);
		else if(op.getOperatorType() == CountDistinctOperatorTypes.KMV)
			return new KMVSketch(op).union(arg0, arg1);
		else if(op.getOperatorType() == CountDistinctOperatorTypes.HLL)
			throw new NotImplementedException("Not implemented yet");
		else
			throw new NotImplementedException("Not implemented yet");
	}
}
