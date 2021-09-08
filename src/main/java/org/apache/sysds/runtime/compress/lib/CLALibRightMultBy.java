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
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibRightMultBy {
	// private static final Log LOG = LogFactory.getLog(CLALibRightMultBy.class.getName());

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k,
		boolean allowOverlap) {
		if(m2.isEmpty()) {
			if(ret == null)
				ret = new MatrixBlock(m1.getNumRows(), m2.getNumColumns(), 0);
			else
				ret.reset(m1.getNumRows(), m2.getNumColumns(), 0);
		}
		else {
			if(m2 instanceof CompressedMatrixBlock) {
				CompressedMatrixBlock m2C = (CompressedMatrixBlock) m2;
				m2 = m2C.getUncompressed("Uncompressed right side of right MM");
			}

			ret = rightMultByMatrixOverlapping(m1, m2, ret, k);

			if(ret instanceof CompressedMatrixBlock) {
				if(!allowOverlap) {
					ret = ((CompressedMatrixBlock) ret).getUncompressed("Overlapping not allowed");
				}
				else {
					final double compressedSize = ret.getInMemorySize();
					final double uncompressedSize = MatrixBlock.estimateSizeDenseInMemory(ret.getNumRows(),
						ret.getNumColumns());
					if(compressedSize * 2 > uncompressedSize)
						ret = ((CompressedMatrixBlock) ret).getUncompressed(
							"Overlapping rep to big: " + compressedSize + " vs Uncompressed " + uncompressedSize);
				}
			}
		}

		ret.recomputeNonZeros();

		return ret;
	}

	private static MatrixBlock rightMultByMatrixOverlapping(CompressedMatrixBlock m1, MatrixBlock that, MatrixBlock ret,
		int k) {
		int rl = m1.getNumRows();
		int cl = that.getNumColumns();
		// Create an overlapping compressed Matrix Block.
		ret = new CompressedMatrixBlock(rl, cl);
		CompressedMatrixBlock retC = (CompressedMatrixBlock) ret;
		ret = rightMultByMatrixCompressed(m1.getColGroups(), that, retC, k);
		return ret;
	}

	private static MatrixBlock rightMultByMatrixCompressed(List<AColGroup> colGroups, MatrixBlock that,
		CompressedMatrixBlock ret, int k) {

		List<AColGroup> retCg = new ArrayList<>();
		boolean containsNull = false;
		if(k == 1) {
			for(AColGroup g : colGroups) {
				AColGroup retG = g.rightMultByMatrix(that);
				if(retG != null)
					retCg.add(retG);
				else
					containsNull = true;
			}
		}
		else {
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				List<Callable<AColGroup>> tasks = new ArrayList<>(colGroups.size());
				for(AColGroup g : colGroups)
					tasks.add(new RightMatrixMultTask(g, that));
				for(Future<AColGroup> fg : pool.invokeAll(tasks)) {
					AColGroup g = fg.get();
					if(g != null)
						retCg.add(g);
					else
						containsNull = true;
				}
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}
		ret.allocateColGroupList(retCg);
		if(retCg.size() > 1)
			ret.setOverlapping(true);

		if(containsNull) {
			ColGroupEmpty cge = findEmptyColumnsAndMakeEmptyColGroup(retCg, ret.getNumColumns(), ret.getNumRows());
			if(cge != null)
				retCg.add(cge);
		}
		return ret;
	}

	private static ColGroupEmpty findEmptyColumnsAndMakeEmptyColGroup(List<AColGroup> colGroups, int nCols, int nRows) {
		Set<Integer> emptyColumns = new HashSet<>(nCols);
		for(int i = 0; i < nCols; i++)
			emptyColumns.add(i);

		for(AColGroup g : colGroups)
			for(int c : g.getColIndices())
				emptyColumns.remove(c);

		if(emptyColumns.size() != 0) {
			int[] emptyColumnsFinal = emptyColumns.stream().mapToInt(Integer::intValue).toArray();
			return new ColGroupEmpty(emptyColumnsFinal, nRows);
		}
		else
			return null;
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
				throw new DMLRuntimeException(e);
			}
		}
	}
}
