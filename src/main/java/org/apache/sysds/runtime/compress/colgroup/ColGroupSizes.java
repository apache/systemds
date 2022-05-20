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

package org.apache.sysds.runtime.compress.colgroup;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public interface ColGroupSizes {
	static final Log LOG = LogFactory.getLog(ColGroupSizes.class.getName());

	public static long estimateInMemorySizeGroup(int nrColumns) {
		long size = 16; // Object header
		size += MemoryEstimates.intArrayCost(nrColumns);
		return size;
	}

	public static long estimateInMemorySizeGroupValue(int nrColumns, int nrValues, double tupleSparsity, boolean lossy) {
		long size = estimateInMemorySizeGroup(nrColumns);
		size += 8; // Counts reference
		size += 8; // Reference to Dict.
		size += DictionaryFactory.getInMemorySize(nrValues, nrColumns, tupleSparsity, lossy);
		return size;
	}

	public static long estimateInMemorySizeDDC(int nrCols, int numTuples, int dataLength, double tupleSparsity,
		boolean lossy) {
		long size = estimateInMemorySizeGroupValue(nrCols, numTuples, tupleSparsity, lossy);
		size += MapToFactory.estimateInMemorySize(dataLength, numTuples);
		return size;
	}

	public static long estimateInMemorySizeOffset(int nrColumns, int nrValues, int pointers, int offsetLength,
		double tupleSparsity, boolean lossy) {
		long size = estimateInMemorySizeGroupValue(nrColumns, nrValues, tupleSparsity, lossy);
		size += 4; // Int nRows
		size += 1; // _zeros boolean reference
		size += 3; // padding
		size += MemoryEstimates.intArrayCost(pointers);
		size += MemoryEstimates.charArrayCost(offsetLength);
		return size;
	}

	public static long estimateInMemorySizeOLE(int nrColumns, int nrValues, int offsetLength, int nrRows,
		double tupleSparsity, boolean lossy) {
		nrColumns = nrColumns > 0 ? nrColumns : 1;
		offsetLength += (nrRows / CompressionSettings.BITMAP_BLOCK_SZ) * 2;
		long size = estimateInMemorySizeOffset(nrColumns, nrValues, nrValues + 1, offsetLength, tupleSparsity, lossy);
		return size;
	}

	public static long estimateInMemorySizeRLE(int nrColumns, int nrValues, int nrRuns, int nrRows, double tupleSparsity,
		boolean lossy) {
		// Correct low number of runs if very large input.
		// This correction handles the case where the skip runs are added in a safe manner
		if(nrRows > Character.MAX_VALUE ) {

			final double extra = (double)nrRows / Character.MAX_VALUE;
			// we assume that half unique values contain extra runs if we have few runs to begin with.
			// This is not 100% guaranteeing larger estimate than real but most likely 
			nrRuns += (extra / 2) * nrValues;
		}
		return estimateInMemorySizeOffset(nrColumns, nrValues, nrValues + 1, nrRuns * 2, tupleSparsity, lossy);
	}

	public static long estimateInMemorySizeSDC(int nrColumns, int nrValues, int nrRows, int largestOff,
		double tupleSparsity, boolean largestOffZero, boolean lossy) {
		long size = estimateInMemorySizeGroupValue(nrColumns, nrValues, tupleSparsity, lossy);
		size += OffsetFactory.estimateInMemorySize(nrRows - largestOff, nrRows);
		if(nrValues > 1 + (largestOffZero ? 0 : 1))
			size += MapToFactory.estimateInMemorySize(nrRows - largestOff, nrValues);
		return size;
	}

	public static long estimateInMemorySizeCONST(int nrColumns, double tupleSparsity, boolean lossy) {
		long size = estimateInMemorySizeGroup(nrColumns);
		size += DictionaryFactory.getInMemorySize(1, nrColumns, tupleSparsity, lossy);
		size += 8; // reference to dictionary.
		return size;
	}

	public static long estimateInMemorySizeEMPTY(int nrColumns) {
		return estimateInMemorySizeGroup(nrColumns);
	}

	public static long estimateInMemorySizeUncompressed(int nrRows, int nrColumns, double sparsity) {
		long size = 0;
		// Since the Object is a col group the overhead from the Memory Size group is added
		size += estimateInMemorySizeGroup(nrColumns);
		size += 8; // reference to MatrixBlock.
		size += MatrixBlock.estimateSizeInMemory(nrRows, nrColumns, (nrColumns > 1) ? sparsity : 1);
		return size;
	}

	public static long estimateInMemorySizeLinearFunctional(int nrColumns) {
		long size = 0;
		// Since the Object is a col group the overhead from the Memory Size group is added
		size += estimateInMemorySizeGroup(nrColumns);
		size += MemoryEstimates.doubleArrayCost(2L * nrColumns); // coefficients; per column, we store 2 doubles (slope &
																					// intercept)
		size += 4; // _numRows
		return size;
	}
}
