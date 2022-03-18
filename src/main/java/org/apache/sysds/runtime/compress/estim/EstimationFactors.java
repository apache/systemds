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
	/** The Number of values in the collection not Zero , Also refered to as singletons */
	protected final int numSingle;
	/** The Number of rows in the column group */
	protected final int numRows;
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

	public EstimationFactors(int nCols, int numVals, int numRows) {
		this.numVals = numVals;
		this.numRows = numRows;
		this.frequencies = null;
		this.numOffs = -1;
		this.largestOff = -1;
		// this.numRuns = -1;
		this.numSingle = -1;
		this.lossy = false;
		this.zeroIsMostFrequent = false;
		this.containNoZeroValues = false;
		this.overAllSparsity = 1;
		this.tupleSparsity = 1;
	}

	public EstimationFactors(int nCols, int numVals, int numOffs, int largestOff, int[] frequencies, int numSingle,
		int numRows, boolean lossy, boolean zeroIsMostFrequent, double overAllSparsity, double tupleSparsity) {
		this.numVals = numVals;
		this.numOffs = numOffs;
		this.largestOff = largestOff;
		this.frequencies = frequencies;
		// this.numRuns = numRuns;
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
		if(numVals <= 0)
			throw new DMLCompressionException("Should not use this constructor if empty");
		if(numOffs <= 0)
			throw new DMLCompressionException("Num offs are to low for this constructor");
		if(numVals > numOffs)
			throw new DMLCompressionException("Num vals cannot be greater than num offs");
		if(largestOff < 0)
			throw new DMLCompressionException("Invalid number of offset, should be greater than one");
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("rows:" + numRows);
		sb.append(" num Offsets:" + numOffs);
		sb.append(" LargestOffset:" + largestOff);
		sb.append(" num Singles:" + numSingle);
		// sb.append(" num Runs:" + numRuns);
		sb.append(" num Unique Vals:" + numVals);
		sb.append(" overallSparsity:" + overAllSparsity);
		sb.append(" tupleSparsity:" + tupleSparsity);
		return sb.toString();
	}
}
