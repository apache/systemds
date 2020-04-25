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

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Logger;
import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.compress.UncompressedBitmap;

/**
 * Compressed Size Estimation factors. Contains meta information used to estimate the compression sizes of given columns
 * into given CompressionFormats
 */
public class CompressedSizeEstimationFactors implements Comparable<CompressedSizeEstimationFactors> {
	static {
		// Set to avoid constructing multiple main loggers.
		Logger.getLogger("org.apache.sysds.runtime.compress.estim");
	}

	protected static final Log LOG = LogFactory.getLog(CompressedSizeEstimationFactors.class.getName());

	protected final int numCols; // Number of columns in the compressed group
	protected final int numVals; // Number of unique values in the compressed group
	protected final int numOffs; // num OLE offsets
	protected final int numRuns; // num RLE runs
	protected final int numSingle; // num singletons
	protected final int numRows;
	protected final boolean containsZero;

	protected CompressedSizeEstimationFactors(int numCols, int numVals, int numOffs, int numRuns, int numSingle,
		int numRows, boolean containsZero) {
		this.numCols = numCols;
		this.numVals = numVals;
		this.numOffs = numOffs;
		this.numRuns = numRuns;
		this.numSingle = numSingle;
		this.numRows = numRows;
		this.containsZero = containsZero;
		LOG.debug(this);
	}

	protected static CompressedSizeEstimationFactors computeSizeEstimationFactors(UncompressedBitmap ubm,
		boolean inclRLE, int numRows, int numCols) {

		int numVals = ubm.getNumValues();

		// TODO: fix the UncompressedBitmap to contain information of if the specific columns extracted
		// contains zero values.
		// This is still not contained in the list because default behavior is to ignore 0 values.
		boolean containsZero = false;

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
				numRuns += list[listSize - 1] / (BitmapEncoder.BITMAP_BLOCK_SZ - 1);
				for(int j = 0; j < listSize; j++) {
					if(list[j] != lastOff + 1) {
						numRuns++;
					}
					lastOff = list[j];
				}
			}
		}

		return new CompressedSizeEstimationFactors(numCols, numVals * numCols, numOffs + numVals, numRuns, numSingle,
			numRows, containsZero);
	}

	protected Iterable<Integer> fieldIterator() {
		ArrayList<Integer> fields = new ArrayList<>();
		fields.add(new Integer(numCols));
		fields.add(numVals);
		fields.add(numOffs);
		fields.add(numRuns);
		fields.add(numSingle);
		fields.add(numRows);
		fields.add(containsZero ? 1 : 0);
		return fields;
	}

	public int compareTo(CompressedSizeEstimationFactors that) {
		int diff = 0;
		Iterator<Integer> thisF = this.fieldIterator().iterator();
		Iterator<Integer> thatF = that.fieldIterator().iterator();

		while(thisF.hasNext() && thatF.hasNext()) {
			Integer thisV = thisF.next();
			Integer thatV = thatF.next();

			if(thisV == thatV) {
				diff = diff << 1;
			}
			else if(thisV > thatV) {
				diff = diff + 1 << 1;
			}
			else {
				diff = diff - 1 << 1;
			}
		}
		return diff;
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