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
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;

/**
 * A helper reusable object for maintaining information about estimated compression
 */
public class CompressedSizeInfo {

	protected static final Log LOG = LogFactory.getLog(CompressedSizeInfo.class.getName());

	public List<CompressedSizeInfoColGroup> compressionInfo;

	public CompressedSizeInfo(CompressedSizeInfoColGroup g){
		this.compressionInfo = new ArrayList<>();
		compressionInfo.add(g);
	}

	public CompressedSizeInfo(List<CompressedSizeInfoColGroup> compressionInfo) {
		this.compressionInfo = compressionInfo;
	}

	public CompressedSizeInfoColGroup getGroupInfo(int index) {
		return compressionInfo.get(index);
	}

	public List<CompressedSizeInfoColGroup> getInfo() {
		return compressionInfo;
	}

	public CompressedSizeInfo setInfo(List<CompressedSizeInfoColGroup> info) {
		compressionInfo = info;
		return this;
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
		for(CompressedSizeInfoColGroup csi : compressionInfo)
			est += csi.getMinSize();

		return est;
	}

	public int getNumberColGroups() {
		return compressionInfo.size();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompressedSizeInfo");
		// sb.append("num Rows" + numRows + " NumCols" + numCols );
		for(CompressedSizeInfoColGroup g : compressionInfo)
			sb.append("\n" + g.toString());
		return sb.toString();
	}

	public String getEstimatedDistinct() {
		StringBuilder sb = new StringBuilder();
		if(compressionInfo == null)
			return "";
		sb.append("[");
		sb.append(compressionInfo.get(0).getNumVals());
		for(int i = 1; i < compressionInfo.size(); i++)
			sb.append(", " + compressionInfo.get(i).getNumVals());
		sb.append("]");
		return sb.toString();
	}

	public String getNrColumnsString() {
		StringBuilder sb = new StringBuilder();
		if(compressionInfo == null)
			return "";
		sb.append("[");
		sb.append(compressionInfo.get(0).getColumns().length);
		for(int i = 1; i < compressionInfo.size(); i++)
			sb.append(", " + compressionInfo.get(i).getColumns().length);
		sb.append("]");
		return sb.toString();
	}
}
