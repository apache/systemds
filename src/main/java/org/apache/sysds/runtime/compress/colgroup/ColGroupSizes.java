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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class ColGroupSizes {
	// ------------------------------
	// Logging parameters:
	// local debug flag
	private static final boolean LOCAL_DEBUG = false;
	// DEBUG/TRACE for details
	private static final Level LOCAL_DEBUG_LEVEL = Level.DEBUG;

	protected static final Log LOG = LogFactory.getLog(ColGroupSizes.class.getName());

	static {
		// for internal debugging only
		if(LOCAL_DEBUG) {
			Logger.getLogger("org.apache.sysds.runtime.compress.colgroup").setLevel(LOCAL_DEBUG_LEVEL);
		}
	}
	// ------------------------------

	public static long getEmptyMemoryFootprint(Class<?> colGroupClass) {
		switch(colGroupClass.getSimpleName()) {
			case "ColGroup":
				return estimateInMemorySizeGroup(0);
			case "ColGroupValue":
				return estimateInMemorySizeGroupValue(0, 0);
			case "ColGroupOffset":
				return estimateInMemorySizeOffset(0, 0, 0, 0);
			case "ColGroupDDC":
				return estimateInMemorySizeDDC(0, 0);
			case "ColGroupDDC1":
				return estimateInMemorySizeDDC1(0, 0, 0);
			case "ColGroupDDC2":
				return estimateInMemorySizeDDC2(0, 0, 0);
			case "ColGroupOLE":
				return estimateInMemorySizeOLE(0, 0, 0, 0);
			case "ColGroupRLE":
				return estimateInMemorySizeRLE(0, 0, 0, 0);
			case "ColGroupUncompressed":
				return estimateInMemorySizeUncompressed(0, 0, 0.0);
			default:
				throw new NotImplementedException("Case not implemented");
		}
	}

	public static long estimateInMemorySizeGroup(int nrColumns) {
		long size = 0;
		size += 16; // Object header
		size += 4; // int numRows,
		size += 1; // _zeros boolean reference
		size += 3; // padding
		size += MemoryEstimates.intArrayCost(nrColumns);
		return size;
	}

	public static long estimateInMemorySizeGroupValue(int nrColumns, long nrValues) {
		long size = estimateInMemorySizeGroup(nrColumns);
		size += 24 //dictionary object
			+ MemoryEstimates.doubleArrayCost(nrValues);
		return size;
	}

	public static long estimateInMemorySizeDDC(int nrCols, int uniqueVals) {
		long size = estimateInMemorySizeGroupValue(nrCols, uniqueVals);
		return size;
	}

	public static long estimateInMemorySizeDDC1(int nrCols, int uniqueVals, int dataLength) {
		if(uniqueVals > 255)
			return Long.MAX_VALUE;
		// LOG.debug("DD1C: " + nrCols + " nr unique: " + uniqueVals + " DataLength: " + dataLength);
		long size = estimateInMemorySizeDDC(nrCols, uniqueVals);
		size += MemoryEstimates.byteArrayCost(dataLength);
		return size;
	}

	public static long estimateInMemorySizeDDC2(int nrCols, int uniqueVals, int dataLength) {
		if(uniqueVals > Character.MAX_VALUE)
			return Long.MAX_VALUE;
		// LOG.debug("DD2C: " + nrCols + "nr unique: " + uniqueVals +" datalen: "+ dataLength);
		long size = estimateInMemorySizeDDC(nrCols, uniqueVals);
		size += MemoryEstimates.charArrayCost(dataLength);
		return size;
	}

	public static long estimateInMemorySizeOffset(int nrColumns, long nrValues, int pointers, int offsetLength) {
		// LOG.debug("OFFSET list: nrC " + nrColumns +"\tnrV " + nrValues + "\tpl "+pointers +"\tdl "+ offsetLength);
		long size = estimateInMemorySizeGroupValue(nrColumns, nrValues);
		size += MemoryEstimates.intArrayCost(pointers);
		size += MemoryEstimates.charArrayCost(offsetLength);
		return size;
	}

	public static long estimateInMemorySizeOLE(int nrColumns, int nrValues, int offsetLength, int nrRows) {
		nrColumns = nrColumns > 0 ? nrColumns : 1;
		offsetLength += (nrRows / BitmapEncoder.BITMAP_BLOCK_SZ) * 2;
		long size = 0;
		// LOG.debug("OLE cols: " + nrColumns + " vals: " + nrValues + " pointers: " + (nrValues / nrColumns + 1)
		// + " offsetLength: " + (offsetLength) + " runs: " + nrValues / nrColumns);
		size = estimateInMemorySizeOffset(nrColumns, nrValues, (nrValues / nrColumns) + 1, offsetLength);
		size += MemoryEstimates.intArrayCost((int) nrValues / nrColumns);
		return size;
	}

	public static long estimateInMemorySizeRLE(int nrColumns, int nrValues, int nrRuns, int nrRows) {
		nrColumns = nrColumns > 0 ? nrColumns : 1;
		int offsetLength = (nrRuns) * 2;
		// LOG.debug("\n\tRLE cols: " + nrColumns + " vals: " + nrValues + " offsetLength: " + offsetLength);
		long size = estimateInMemorySizeOffset(nrColumns, nrValues, (nrValues / nrColumns) + 1, offsetLength);

		return size;
	}

	public static long estimateInMemorySizeUncompressed(int nrRows, int nrColumns, double sparsity) {
		long size = 0;
		// Since the Object is a col group the overhead from the Memory Size group is added
		size += estimateInMemorySizeGroup(nrColumns);
		size += 8; // reference to MatrixBlock.
		size += MatrixBlock.estimateSizeInMemory(nrRows, nrColumns, sparsity);
		return size;
	}
}