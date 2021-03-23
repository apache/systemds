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

package org.apache.sysds.runtime.compress.cocode;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public abstract class AColumnGroupPartitioner {

	protected static final Log LOG = LogFactory.getLog(AColumnGroupPartitioner.class.getName());

	protected CompressedSizeEstimator _est;
	protected CompressionSettings _cs;
	protected int _numRows;

	protected AColumnGroupPartitioner(CompressedSizeEstimator sizeEstimator, CompressionSettings cs, int numRows) {
		_est = sizeEstimator;
		_cs = cs;
		_numRows = numRows;
	}

	public abstract CompressedSizeInfo partitionColumns(CompressedSizeInfo colInfos);

	protected CompressedSizeInfoColGroup join(CompressedSizeInfoColGroup lhs, CompressedSizeInfoColGroup rhs, boolean analyze){
		return analyze ? joinWithAnalysis(lhs, rhs): joinWithoutAnalysis(lhs,rhs);
	}

	protected CompressedSizeInfoColGroup joinWithAnalysis(CompressedSizeInfoColGroup lhs, CompressedSizeInfoColGroup rhs) {
		int[] joined = join(lhs.getColumns(), rhs.getColumns());
		return _est.estimateCompressedColGroupSize(joined);
	}

	protected CompressedSizeInfoColGroup joinWithoutAnalysis(CompressedSizeInfoColGroup lhs, CompressedSizeInfoColGroup rhs){
		int[] joined = join(lhs.getColumns(), rhs.getColumns());
		int numVals =  lhs.getNumVals() + rhs.getNumVals();
		return new CompressedSizeInfoColGroup(joined, numVals, _numRows);
	}

	protected CompressedSizeInfoColGroup analyze(CompressedSizeInfoColGroup g){
		if(g.getBestCompressionType() == null)
			return _est.estimateCompressedColGroupSize(g.getColumns());
		else 
			return g;
	}

	private int[] join(int[] lhs, int[] rhs){
		int[] joined = new int[lhs.length + rhs.length];
		int lp = 0;
		int rp = 0;
		int i = 0;
		for(; i < joined.length && lp < lhs.length && rp < rhs.length; i++){
			if(lhs[lp] < rhs[rp]){
				joined[i] = lhs[lp++];
			}else{
				joined[i] = rhs[rp++];
			}
		}

		while(lp < lhs.length){
			joined[i++] = lhs[lp++];
		}
		while(rp < rhs.length){
			joined[i++] = rhs[rp++];
		}
		return joined;
	}
}
