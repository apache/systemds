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
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;

/**
 * A helper reusable object for maintaining information about estimated compression
 */
public class CompressedSizeInfo {
	
	protected static final Log LOG = LogFactory.getLog(CompressedSizeInfo.class.getName());

	public CompressedSizeInfoColGroup[] compressionInfo;
	public int numCols;
	public int nnz;
	public int numRows;

	public CompressedSizeInfo(CompressedSizeInfoColGroup[] compressionInfo, int nnz, int numRows, int numCols) {
		if(numCols < 0 )
			throw new DMLCompressionException("Invalid number of columns");
		this.compressionInfo = compressionInfo;
		this.nnz = nnz;
		this.numRows = numRows;
		this.numCols = numCols;
	}

	public int[][] getGroups(){
		int[][] ret = new int[compressionInfo.length][];
		for(int i = 0; i < compressionInfo.length; i++){
			ret[i] = compressionInfo[i].getColumns();
		}
		return ret;
	}

	public CompressedSizeInfoColGroup getGroupInfo(int index) {
		return compressionInfo[index];
	}

	public CompressedSizeInfoColGroup[] getInfo(){
		return compressionInfo;
	}

	public void setInfo(CompressedSizeInfoColGroup[] info){
		compressionInfo = info;
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

	public int getNumberColGroups(){
		return compressionInfo.length;
	}

	public boolean isCompressible(long orgSize) {
		long sum = 0;
		for(CompressedSizeInfoColGroup g : compressionInfo) {
			sum += g.getMinSize();
		}
		// LOG.error("Original size :" + orgSize + "compressedSingleColumns: " + sum);
		return sum <= orgSize;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompressedSizeInfo");
		sb.append("num Rows" + numRows + "  NumCols" + numCols + "  nnz" + nnz);
		for(CompressedSizeInfoColGroup g : compressionInfo)
			sb.append("\n" + g.toString());
		return sb.toString();
	}
}
