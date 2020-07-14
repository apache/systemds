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

import java.util.HashMap;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;

/**
 * A helper reusable object for maintaining information about estimated compression
 */
public class CompressedSizeInfo {

	public CompressedSizeInfoColGroup[] compressionInfo;
	public List<Integer> colsC;
	public List<Integer> colsUC;
	public HashMap<Integer, Double> compRatios;
	public int nnzUC;

	public CompressedSizeInfo(CompressedSizeInfoColGroup[] compressionInfo, List<Integer> colsC, List<Integer> colsUC,
		HashMap<Integer, Double> compRatios, int nnzUC) {
		this.compressionInfo = compressionInfo;
		this.colsC = colsC;
		this.colsUC = colsUC;
		this.compRatios = compRatios;
		this.nnzUC = nnzUC;
	}

	public CompressedSizeInfoColGroup getGroupInfo(int index) {
		return compressionInfo[index];
	}

	/**
	 * Method for returning the calculated memory usage from this specific compression plan.
	 * 
	 * @return The in memory estimate as a long counting bytes.
	 */
	public long memoryEstimate() {
		// Basic data inherited from MatrixBlock + CompressedMatrixBlock
		long est = CompressedMatrixBlock.baseSizeInMemory();
		// Memory usage from all Compression Groups.
		for(CompressedSizeInfoColGroup csi : compressionInfo) {
			est += csi.getMinSize();
		}

		return est;
	}

}
