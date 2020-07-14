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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.ABitmap.BitmapType;

/**
 * Compressed Size Estimation factors. Contains meta information used to estimate the compression sizes of given columns
 * into given CompressionFormats
 */
public class EstimationFactors {

	protected static final Log LOG = LogFactory.getLog(EstimationFactors.class.getName());

	protected final int numCols; // Number of columns in the compressed group
	// TODO Make a variable called numDistinct to use for DDC.
	/** Number of distinct value tuples in the columns, not to be confused with number of distinct values */
	protected final int numVals; // Number of unique values in the compressed group
	/** The number of offsets, to tuples of values in the column groups */
	protected final int numOffs;
	/** The Number of runs, of consecutive equal numbers, used primarily in RLE */
	protected final int numRuns;
	/** The Number of Values in the collection not Zero , Also refered to as singletons */
	protected final int numSingle;
	protected final int numRows;
	protected final boolean containsZero;
	protected final boolean lossy;

	protected EstimationFactors(int numCols, int numVals, int numOffs, int numRuns, int numSingle, int numRows,
		boolean containsZero, boolean lossy) {
		this.numCols = numCols;
		this.numVals = numVals;
		this.numOffs = numOffs;
		this.numRuns = numRuns;
		this.numSingle = numSingle;
		this.numRows = numRows;
		this.containsZero = containsZero;
		this.lossy = lossy;
		LOG.debug(this);
	}

	protected static EstimationFactors computeSizeEstimationFactors(ABitmap ubm, boolean inclRLE, int numRows,
		int numCols) {
		int numVals = ubm.getNumValues();
		boolean containsZero = ubm.containsZero();

		int numRuns = 0;
		int numOffs = 0;
		int numSingle = 0;

		LOG.debug("NumCols :" + numCols);

		// compute size estimation factors
		for(int i = 0; i < numVals; i++) {
			int listSize = ubm.getNumOffsets(i);
			numOffs += listSize;
			numSingle += (listSize == 1) ? 1 : 0;
			if(inclRLE) {
				int[] list = ubm.getOffsetsList(i).extractValues();
				int lastOff = -2;
				numRuns += list[listSize - 1] / (CompressionSettings.BITMAP_BLOCK_SZ - 1);
				for(int j = 0; j < listSize; j++) {
					if(list[j] != lastOff + 1) {
						numRuns++;
					}
					lastOff = list[j];
				}
			}
		}

		return new EstimationFactors(numCols, numVals * numCols, numOffs + numVals, numRuns, numSingle, numRows,
			containsZero, ubm.getType() == BitmapType.Lossy);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nrows:" + numRows);
		sb.append("\tcols:" + numCols);
		sb.append("\tnum Offsets:" + numOffs);
		sb.append("\tnum Singles:" + numSingle);
		sb.append("\tnum Runs:" + numRuns);
		sb.append("\tnum Unique Vals:" + numVals);
		sb.append("\tcontains a 0: " + containsZero);
		return sb.toString();
	}
}