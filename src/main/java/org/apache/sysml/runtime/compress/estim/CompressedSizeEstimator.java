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

package org.apache.sysml.runtime.compress.estim;

import org.apache.sysml.runtime.compress.BitmapEncoder;
import org.apache.sysml.runtime.compress.UncompressedBitmap;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Base class for all compressed size estimators
 */
public abstract class CompressedSizeEstimator 
{
	protected MatrixBlock _data;

	public CompressedSizeEstimator(MatrixBlock data) {
		_data = data;
	}

	/**
	 * 
	 * @param colIndexes
	 * @return
	 */
	public abstract CompressedSizeInfo estimateCompressedColGroupSize(int[] colIndexes);

	/**
	 * 
	 * @param ubm
	 * @return
	 */
	public abstract CompressedSizeInfo estimateCompressedColGroupSize(UncompressedBitmap ubm);

	/**
	 * 
	 * @param ubm
	 * @param inclRLE
	 * @return
	 */
	protected SizeEstimationFactors computeSizeEstimationFactors(UncompressedBitmap ubm, boolean inclRLE) {
		int numVals = ubm.getNumValues();
		int numRuns = 0;
		int numOffs = 0;
		int numSegs = 0;
		int numSingle = 0;
		
		//compute size estimation factors
		for (int i = 0; i < numVals; i++) {
			int[] list = ubm.getOffsetsList(i);
			numOffs += list.length;
			numSegs += list[list.length - 1] / BitmapEncoder.BITMAP_BLOCK_SZ + 1;
			numSingle += (list.length==1) ? 1 : 0;
			if( inclRLE ) {
				int lastOff = -2;
				for (int j = 0; j < list.length; j++) {
					if (list[j] != lastOff + 1)
						numRuns++;
					lastOff = list[j];
				}
			}
		}
		
		//construct estimation factors
		return new SizeEstimationFactors(numVals, numSegs, numOffs, numRuns, numSingle);
	}

	/**
	 * Estimates the number of bytes needed to encode this column group
	 * in RLE encoding format.
	 * 
	 * @param numVals
	 * @param numRuns
	 * @param numCols
	 * @return
	 */
	protected static long getRLESize(int numVals, int numRuns, int numCols) {
		int ret = 0;
		//distinct value tuples [double per col]
		ret += 8 * numVals * numCols;
		//offset/len fields per distinct value tuple [2xint]
		ret += 8 * numVals;
		//run data [2xchar]
		ret += 4 * numRuns;
		return ret;
	}

	/**
	 * Estimates the number of bytes needed to encode this column group 
	 * in OLE format.
	 * 
	 * @param numVals
	 * @param numOffs
	 * @param numSeqs
	 * @param numCols
	 * @return
	 */
	protected static long getOLESize(int numVals, float numOffs, int numSeqs, int numCols) {
		int ret = 0;
		//distinct value tuples [double per col]
		ret += 8 * numVals * numCols;
		//offset/len fields per distinct value tuple [2xint]
		ret += 8 * numVals;
		//offset list data [1xchar]
		ret += 2 * numOffs;
		//offset list seqment headers [1xchar]
		ret += 2 * numSeqs;
		return ret;
	}
	
	/**
	 * 
	 */
	protected static class SizeEstimationFactors {
 		protected int numVals;   //num value tuples
 		protected int numSegs;   //num OLE segments 
 		protected int numOffs;   //num OLE offsets
 		protected int numRuns;   //num RLE runs
 		protected int numSingle; //num singletons
		
		protected SizeEstimationFactors(int numvals, int numsegs, int numoffs, int numruns, int numsingle) {
			numVals = numvals;
			numSegs = numsegs;
			numOffs = numoffs;
			numRuns = numruns;
			numSingle = numsingle;
		}
	}
}
