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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;

/**
 * Compressed Size Estimation factors. Contains meta information used to estimate the compression sizes of given columns
 * into given CompressionFormats
 */
public class EstimationFactors {

	protected static final Log LOG = LogFactory.getLog(EstimationFactors.class.getName());

	/** Number of distinct value tuples in the columns, not to be confused with number of distinct values */
	protected final int numVals;
	/** The number of offsets, to tuples of values in the column groups */
	protected final int numOffs;
	/** The number of instances in the largest offset, this is used to determine if SDC is good. */
	protected final int largestOff;
	/** The frequencies of the Non zero tuples in the columns */
	protected final int[] frequencies;
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

	protected EstimationFactors(int nCols, int numVals, int numRows) {
		// Safety in numbers, if the estimation factor is saying that there is no values in 5 columns,
		// then add one to make sure that the cocode does not run amok. If the data is actually empty, the columns
		// are joined later.
		if(nCols >= 5 && numVals == 0)
			this.numVals = 1;
		else
			this.numVals = numVals;
		this.numRows = numRows;
		this.frequencies = null;
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

	protected EstimationFactors(int nCols, int numVals, int numRows, int largestOff, double sparsity) {
		// Safety in numbers, if the estimation factor is saying that there is no values in 5 columns,
		// then add one to make sure that the cocode does not run amok. If the data is actually empty, the columns
		// are joined later.
		if(nCols >= 5 && numVals == 0)
			this.numVals = 1;
		else
			this.numVals = numVals;
		this.numRows = numRows;
		this.frequencies = null;
		this.numOffs = (int) (numRows * sparsity);
		this.largestOff = largestOff;
		this.numRuns = -1;
		this.numSingle = -1;
		this.lossy = false;
		this.zeroIsMostFrequent = true;
		this.containNoZeroValues = false;
		this.overAllSparsity = sparsity;
		this.tupleSparsity = 1;
	}

	protected EstimationFactors(int nCols, EstimationFactors old) {
		if(nCols >= 5 && old.numVals == 0)
			this.numVals = 1;
		else
			this.numVals = old.numVals;
		this.numRows = old.numRows;
		this.numOffs = old.numOffs;
		this.largestOff = old.largestOff;
		this.frequencies = old.frequencies;
		this.numRuns = old.numRuns;
		this.numSingle = old.numSingle;
		this.lossy = old.lossy;
		this.zeroIsMostFrequent = old.zeroIsMostFrequent;
		this.containNoZeroValues = old.containNoZeroValues;
		this.overAllSparsity = old.overAllSparsity;
		this.tupleSparsity = old.tupleSparsity;
	}

	protected EstimationFactors(int nCols, int numVals, int numOffs, int largestOff, int[] frequencies, int numRuns,
		int numSingle, int numRows, boolean lossy, boolean zeroIsMostFrequent, double overAllSparsity,
		double tupleSparsity) {
		// Safety in numbers, if the estimation factor is saying that there is no values in 5 columns,
		// then add one to make sure that the cocode does not run amok. If the data is actually empty, the columns
		// are joined later.
		if(nCols >= 5 && numVals == 0)
			this.numVals = 1;
		else
			this.numVals = numVals;
		this.numOffs = numOffs;
		this.largestOff = largestOff;
		this.frequencies = frequencies;
		this.numRuns = numRuns;
		this.numSingle = numSingle;
		this.numRows = numRows;
		this.lossy = lossy;
		this.zeroIsMostFrequent = zeroIsMostFrequent;
		this.containNoZeroValues = numOffs == numRows && overAllSparsity < 1;
		this.overAllSparsity = overAllSparsity;
		this.tupleSparsity = tupleSparsity;
			
		if(overAllSparsity > 1 || overAllSparsity < 0)
			throw new DMLCompressionException("Invalid OverAllSparsity of: " + overAllSparsity);
		if(tupleSparsity > 1 || tupleSparsity < 0)
			throw new DMLCompressionException("Invalid TupleSparsity of:" + tupleSparsity);
		if(largestOff > numRows)
			throw new DMLCompressionException(
				"Invalid number of instance of most common element should be lower than number of rows. " + largestOff
					+ " > numRows: " + numRows);
	}

	protected static EstimationFactors emptyFactors(int nCols, int nRows){
		return new EstimationFactors(nCols, 0, 0, 0 , null, 0, 0, nRows, false, true, 0, 0);
	}

	protected static EstimationFactors computeSizeEstimationFactors(ABitmap ubm, int rlen, boolean inclRLE, int[] cols) {
		if(ubm == null || ubm.getOffsetList() == null)
			return null;

		int numVals = ubm.getNumValues();
		int numRuns = 0;
		int numOffs = 0;
		int numSingle = 0;
		int largestOffs = 0;
		long tupleNonZeroCount = 0;
		long overallNonZeroCount = 0;
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

		final int zerosOffs = rlen - numOffs;
		int[] frequencies = new int[numVals];
		for(int i = 0; i < numVals; i++)
			frequencies[i] = ubm.getNumOffsets(i);
		final boolean zerosLargestOffset = zerosOffs > largestOffs;
		if(zerosLargestOffset)
			largestOffs = zerosOffs;

		double overAllSparsity = (double) overallNonZeroCount / (rlen * cols.length);
		double tupleSparsity = (double) tupleNonZeroCount / (numVals * cols.length);

		return new EstimationFactors(cols.length, numVals, numOffs, largestOffs, frequencies, numRuns, numSingle, rlen,
			false, zerosLargestOffset, overAllSparsity, tupleSparsity);

	}

	protected static EstimationFactors computeSizeEstimation(final int[] cols, final AMapToData map,
		final boolean inclRLE, final int numRows, final boolean lastEntryAllZero) {
		final boolean lossy = false;
		if(map == null)
			return null;

		final int nUnique = map.getUnique();
		if(lastEntryAllZero || inclRLE)
			throw new NotImplementedException();

		final boolean zerosLargestOffset = false;
		final double overAllSparsity = 1.0;
		final double tupleSparsity = 1.0;
		final int numOffs = map.size();
		final int numSingle = 0; // unknown

		final int numRuns = 0;
		int[] counts = new int[nUnique];
		for(int i = 0; i < map.size(); i++)
			counts[map.getIndex(i)]++;

		int largestOffs = 0;
		for(int i = 0; i < nUnique; i++)
			if(counts[i] > largestOffs)
				largestOffs = counts[i];

		return new EstimationFactors(cols.length, nUnique, numOffs, largestOffs, counts, numRuns, numSingle, numRows,
			lossy, zerosLargestOffset, overAllSparsity, tupleSparsity);

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
		return sb.toString();
	}
}
