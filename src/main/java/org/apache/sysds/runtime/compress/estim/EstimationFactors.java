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

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;

/**
 * Compressed Size Estimation factors. Contains meta information used to estimate the compression sizes of given columns
 * into given CompressionFormats
 */
public class EstimationFactors {

	/** Number of distinct value tuples in the columns, not to be confused with number of distinct values */
	protected final int numVals;
	/** The number of offsets, to tuples of values in the column groups */
	protected final int numOffs;
	/** The number of instances in the largest offset, this is used to determine if SDC is good. */
	protected final int largestOff;
	/** The frequencies of the Non zero tuples in the columns */
	protected final int[] frequencies;
	/** The Number of values in the collection not Zero, also referred to as singletons */
	protected final int numSingle;
	/** The Number of rows in the column group */
	protected final int numRows;
	/** The Number of runs of continuous values inside the column group */
	protected final int numRuns;
	/** If the estimation of this column group is lossy */
	protected final boolean lossy;
	/** Boolean specifying if zero is the most frequent value */
	protected final boolean zeroIsMostFrequent;
	/** Boolean specifying if the columnGroup contain no zero tuples. */
	protected final boolean containNoZeroValues;
	/** The sparsity of the overall Factors including the number of each distinct tuple. */
	protected final double overAllSparsity;
	/** The sparsity of the tuples them selves in isolation */
	protected final double tupleSparsity;

	public EstimationFactors(int numVals, int numRows) {
		this(numVals, numRows, -1, null, -1, numRows, false, false, 1.0, 1.0);
	}

	public EstimationFactors(int numVals, int numRows, double tupleSparsity) {
		this(numVals, numRows, -1, null, -1, numRows, false, false, 1.0, tupleSparsity);
	}

	public EstimationFactors(int numVals, int numRows, int numOffs, double tupleSparsity) {
		this(numVals, numOffs, -1, null, -1, numRows, false, false, 1.0, tupleSparsity);
	}

	public EstimationFactors(int numVals, int numOffs, int largestOff, int[] frequencies, int numSingle, int numRows,
		boolean lossy, boolean zeroIsMostFrequent, double overAllSparsity, double tupleSparsity) {
		this(numVals, numOffs, largestOff, frequencies, numSingle, numRows, numOffs, lossy, zeroIsMostFrequent,
			overAllSparsity, tupleSparsity);
	}

	public EstimationFactors(int numVals, int numOffs, int largestOff, int[] frequencies, int numSingle, int numRows,
		int numRuns, boolean lossy, boolean zeroIsMostFrequent, double overAllSparsity, double tupleSparsity) {
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
			overAllSparsity = Math.max(0, Math.min(1, overAllSparsity));
		else if(tupleSparsity > 1 || tupleSparsity < 0)
			tupleSparsity = Math.max(0, Math.min(1, tupleSparsity));
		else if(largestOff > numRows)
			largestOff = numRows;
		else if(numVals > numOffs)
			numVals = numOffs;
		

		if(CompressedMatrixBlock.debug && frequencies != null) {
			for(int i = 0; i < frequencies.length; i++) {
				if(frequencies[i] == 0)
					throw new DMLCompressionException("Invalid counts in fact contains 0");
			}
		}
	}

	public int[] getFrequencies() {
		return frequencies;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("rows:" + numRows);
		sb.append(" num Offsets:" + numOffs);
		if(largestOff != -1)
			sb.append(" LargestOffset:" + largestOff);
		sb.append(" num Singles:" + numSingle);
		if(numRuns != numOffs)
			sb.append(" num Runs: " + numRuns);
		sb.append(" num Unique Vals:" + numVals);
		sb.append(" overallSparsity:" + overAllSparsity);
		sb.append(" tupleSparsity:" + tupleSparsity);
		return sb.toString();
	}
}
