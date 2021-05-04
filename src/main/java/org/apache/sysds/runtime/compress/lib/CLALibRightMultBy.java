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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibRightMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibRightMultBy.class.getName());

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k,
		boolean allowOverlap) {
		ret = rightMultByMatrix(m1.getColGroups(), m2, ret, k, m1.getMaxNumValues(), allowOverlap);
		ret.recomputeNonZeros();
		return ret;
	}

	private static MatrixBlock rightMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		Pair<Integer, int[]> v, boolean allowOverlap) {

		if(that instanceof CompressedMatrixBlock)
			LOG.warn("Decompression Right matrix");

		that = that instanceof CompressedMatrixBlock ? ((CompressedMatrixBlock) that).decompress(k) : that;

		MatrixBlock m = rightMultByMatrixOverlapping(colGroups, that, ret, k, v);
		if(m instanceof CompressedMatrixBlock)
			if(allowOverlappingOutput(colGroups, allowOverlap))
				return m;
			else {
				CompressedMatrixBlock outBlock = (CompressedMatrixBlock) m;
				ColGroupUncompressed uccg = findUncompressedColGroup(outBlock.getColGroups());
				if(uccg == null)
					return outBlock.decompress(k);
				else{
					MatrixBlock outputBlock = uccg.getData();
					outputBlock.sparseToDense();
					return outBlock.decompress(outputBlock, k);
				}
			}
		else
			return m;
	}

	private static ColGroupUncompressed findUncompressedColGroup(List<AColGroup> colGroups){
		for(AColGroup g: colGroups){
			if(g instanceof ColGroupUncompressed)
				return (ColGroupUncompressed) g;
		}
		return null;
	}

	private static boolean allowOverlappingOutput(List<AColGroup> colGroups, boolean allowOverlap) {
		
		if(!allowOverlap) {
			LOG.debug("Not Overlapping because it is not allowed");
			return false;
		}
		else
			return true;
		// int distinctCount = 0;
		// for(AColGroup g : colGroups) {
		// 	if(g instanceof ColGroupCompressed)
		// 		distinctCount += ((ColGroupCompressed) g).getNumValues();
		// 	else {
		// 		LOG.debug("Not Overlapping because there is an un-compressed column group");
		// 		return false;
		// 	}
		// }
		// final int threshold = colGroups.get(0).getNumRows() / 2;
		// boolean allow = distinctCount <= threshold;
		// if(LOG.isDebugEnabled() && !allow)
		// 	LOG.debug("Not Allowing Overlap because of number of distinct items in compression: " + distinctCount
		// 		+ " is greater than threshold: " + threshold);
		// return allow;

	}

	private static MatrixBlock rightMultByMatrixOverlapping(List<AColGroup> colGroups, MatrixBlock that,
		MatrixBlock ret, int k, Pair<Integer, int[]> v) {
		int rl = colGroups.get(0).getNumRows();
		int cl = that.getNumColumns();
		// Create an overlapping compressed Matrix Block.
		ret = new CompressedMatrixBlock(rl, cl);
		CompressedMatrixBlock retC = (CompressedMatrixBlock) ret;
		ret = rightMultByMatrixCompressed(colGroups, that, retC, k, v);
		return ret;
	}

	private static MatrixBlock rightMultByMatrixCompressed(List<AColGroup> colGroups, MatrixBlock that,
		CompressedMatrixBlock ret, int k, Pair<Integer, int[]> v) {

		List<AColGroup> retCg = new ArrayList<>();
		if(k == 1) {
			for(AColGroup g : colGroups)
				retCg.add(g.rightMultByMatrix(that));
		}
		else {
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				List<Callable<AColGroup>> tasks = new ArrayList<>(colGroups.size());
				for(AColGroup g : colGroups)
					tasks.add(new RightMatrixMultTask(g, that));
				for(Future<AColGroup> fg : pool.invokeAll(tasks)){
					AColGroup g = fg.get();
					if(g != null)
						retCg.add(g);
				}
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}
		ret.allocateColGroupList(retCg);
		if(retCg.size() > 1)
			ret.setOverlapping(true);
		return ret;
	}

	private static class RightMatrixMultTask implements Callable<AColGroup> {
		private final AColGroup _colGroup;
		private final MatrixBlock _b;

		protected RightMatrixMultTask(AColGroup colGroup, MatrixBlock b) {
			_colGroup = colGroup;
			_b = b;
		}

		@Override
		public AColGroup call() {
			try {
				return _colGroup.rightMultByMatrix(_b);
			}
			catch(Exception e) {
				LOG.error(e);
				throw new DMLRuntimeException(e);
			}
		}
	}
}
