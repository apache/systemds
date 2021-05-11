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

package org.apache.sysds.runtime.compress.estim;

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.ABitmap.BitmapType;

/**
 * Compressed Size Estimation factors. Contains meta information used to estimate the compression sizes of given columns
 * into given CompressionFormats
 */
public class EstimationFactors {

	protected static final Log LOG = LogFactory.getLog(EstimationFactors.class.getName());

	protected final int[] cols;
	/** Number of distinct value tuples in the columns, not to be confused with number of distinct values */
	protected final int numVals;
	/** The number of offsets, to tuples of values in the column groups */
	protected final int numOffs;
	/** The number of instances in the largest offset, this is used to determine if SDC is good. */
	protected final int largestOff;
	/** The Number of runs, of consecutive equal numbers, used primarily in RLE */
	protected final int numRuns;
	/** The Number of Values in the collection not Zero , Also refered to as singletons */
	protected final int numSingle;
	protected final int numRows;
	protected final boolean lossy;
	/** Boolean specifying if zero is the most frequent value */
	protected final boolean zeroIsMostFrequent;
	/** Boolean specifying if the columnGroup contain no zero tuples. */
	protected final boolean containNoZeroValues;
	/** The sparsity of the overall Factors including the number of each distinct tuple. */
	protected final double overAllSparsity;
	/** The sparsity of the tuples them selves in isolation */
	protected final double tupleSparsity;

	protected EstimationFactors(int[] cols, int numVals, int numRows) {
		this.cols = cols;
		this.numVals = numVals;
		this.numRows = numRows;

		this.numOffs = -1;
		this.largestOff = -1;
		this.numRuns = -1;
		this.numSingle = -1;
		this.lossy = false;
		this.zeroIsMostFrequent = false;
		this.containNoZeroValues = false;
		this.overAllSparsity = 1;
		this.tupleSparsity = 1;
	}

	protected EstimationFactors(int[] cols, EstimationFactors old) {
		this.cols = cols;
		this.numVals = old.numVals;
		this.numRows = old.numRows;
		this.numOffs = old.numOffs;
		this.largestOff = old.largestOff;
		this.numRuns = old.numRuns;
		this.numSingle = old.numSingle;
		this.lossy = old.lossy;
		this.zeroIsMostFrequent = old.zeroIsMostFrequent;
		this.containNoZeroValues = old.containNoZeroValues;
		this.overAllSparsity = old.overAllSparsity;
		this.tupleSparsity = old.tupleSparsity;
	}

	protected EstimationFactors(int[] cols, int numVals, int numOffs, int largestOff, int numRuns, int numSingle,
		int numRows, boolean lossy, boolean zeroIsMostFrequent, double overAllSparsity, double tupleSparsity) {
		this.cols = cols;
		this.numVals = numVals;
		this.numOffs = numOffs;
		this.largestOff = largestOff;
		this.numRuns = numRuns;
		this.numSingle = numSingle;
		this.numRows = numRows;
		this.lossy = lossy;
		this.zeroIsMostFrequent = zeroIsMostFrequent;
		this.containNoZeroValues = numOffs == numRows;
		this.overAllSparsity = overAllSparsity;
		this.tupleSparsity = tupleSparsity;

		if(!containNoZeroValues && overAllSparsity >= 1)
			throw new DMLCompressionException(
				"Invalid Sparsity, if there is zeroOffsets, then the sparsity should be below 1");
		if(overAllSparsity > 1 || overAllSparsity < 0)
			throw new DMLCompressionException("Invalid sparsity");
		if(tupleSparsity > 1 || tupleSparsity < 0)
			throw new DMLCompressionException("Invalid sparsity");
		if(largestOff > numRows)
			throw new DMLCompressionException(
				"Invalid number of instance of most common element should be lower than number of rows. " + largestOff
					+ " > numRows: " + numRows);
	}

	protected static EstimationFactors computeSizeEstimationFactors(ABitmap ubm, boolean inclRLE, int numRows,
		int[] cols) {
		if(ubm == null || ubm.getOffsetList() == null)
			return new EstimationFactors(cols, 0, 0, numRows, 1, 0, numRows, false, true, 0, 0);
		else {
			final int numVals = ubm.getNumValues();
			int numRuns = 0;
			int numOffs = 0;
			int numSingle = 0;
			int largestOffs = 0;
			int tupleNonZeroCount = 0;
			int overallNonZeroCount = 0;
			// compute size estimation factors
			for(int i = 0; i < numVals; i++) {
				final int listSize = ubm.getNumOffsets(i);
				final int numZerosInTuple = ubm.getNumNonZerosInOffset(i);
				tupleNonZeroCount += numZerosInTuple;
				overallNonZeroCount += numZerosInTuple * listSize;
				numOffs += listSize;
				if(listSize > largestOffs)
					largestOffs = listSize;

				numSingle += (listSize == 1) ? 1 : 0;
				if(inclRLE) {
					int[] list = ubm.getOffsetsList(i).extractValues();
					int lastOff = -2;
					numRuns += list[listSize - 1] / (CompressionSettings.BITMAP_BLOCK_SZ);
					for(int j = 0; j < listSize; j++) {
						if(list[j] != lastOff + 1)
							numRuns++;
						lastOff = list[j];
					}
				}
			}

			final int zerosOffs = numRows - numOffs;
			final boolean zerosLargestOffset = zerosOffs > largestOffs;
			if(zerosLargestOffset)
				largestOffs = zerosOffs;

			double overAllSparsity = (double) overallNonZeroCount / (numRows * cols.length);
			double tupleSparsity = (double) tupleNonZeroCount / (numVals * cols.length);

			return new EstimationFactors(cols, numVals, numOffs, largestOffs, numRuns, numSingle, numRows,
				ubm.getType() == BitmapType.Lossy, zerosLargestOffset, overAllSparsity, tupleSparsity);
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" rows:" + numRows);
		sb.append(" num Offsets:" + numOffs);
		sb.append(" LargestOffset:" + largestOff);
		sb.append(" num Singles:" + numSingle);
		sb.append(" num Runs:" + numRuns);
		sb.append(" num Unique Vals:" + numVals);
		sb.append(" overallSparsity:" + overAllSparsity);
		sb.append(" tupleSparsity:" + tupleSparsity);
		sb.append(" cols:" + Arrays.toString(cols));
		return sb.toString();
	}
}
