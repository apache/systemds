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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public final class CLALibCBind {

	private CLALibCBind() {
		// private constructor.
	}

	private static final Log LOG = LogFactory.getLog(CLALibCBind.class.getName());

	public static MatrixBlock cbind(MatrixBlock left, MatrixBlock[] right, int k) {
		try {

			if(right.length == 1) {
				return cbind(left, right[0], k);
			}
			else {
				boolean allCompressed = true;
				for(int i = 0; i < right.length && allCompressed; i++)
					allCompressed = right[i] instanceof CompressedMatrixBlock;
				if(allCompressed)
					return cbindAllCompressed((CompressedMatrixBlock) left, right, k);
				else
					return cbindAllNormalCompressed(left, right, k);
			}
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed to Cbind with compressed input", e);
		}
	}

	private static MatrixBlock cbindAllNormalCompressed(MatrixBlock left, MatrixBlock[] right, int k) {
		for(int i = 0; i < right.length; i++) {
			left = cbind(left, right[i], k);
		}
		return left;
	}

	public static MatrixBlock cbind(MatrixBlock left, MatrixBlock right, int k) {

		final int m = left.getNumRows();
		final int n = left.getNumColumns() + right.getNumColumns();

		if(left.isEmpty() && right instanceof CompressedMatrixBlock)
			return appendLeftEmpty(left, (CompressedMatrixBlock) right, m, n);
		else if(right.isEmpty() && left instanceof CompressedMatrixBlock)
			return appendRightEmpty((CompressedMatrixBlock) left, right, m, n);

		if(!(left instanceof CompressedMatrixBlock) && left.getInMemorySize() < 1000) {
			LOG.info("Trying to compress left side of append");
			left = CompressedMatrixBlockFactory.compress(left, k).getLeft();
		}

		if(!(right instanceof CompressedMatrixBlock) && left.getInMemorySize() < 1000) {
			LOG.info("Trying to compress right side of append");
			right = CompressedMatrixBlockFactory.compress(right, k).getLeft();
		}

		// if compression failed then use default append method.
		if(!(left instanceof CompressedMatrixBlock && right instanceof CompressedMatrixBlock)) {
			final double spar = (left.getNonZeros() + right.getNonZeros()) / ((double) m * n);
			final double estSizeUncompressed = MatrixBlock.estimateSizeInMemory(m, n, spar);
			final double estSizeCompressed = left.getInMemorySize() + right.getInMemorySize();
			// if(isAligned((CompressedMatrixBlock) left, (CompressedMatrixBlock) right))
			// return combineCompressed((CompressedMatrixBlock) left, (CompressedMatrixBlock) right);
			// else
			if(estSizeUncompressed < estSizeCompressed)
				return uc(left).append(uc(right), null);
			else if(left instanceof CompressedMatrixBlock)
				return appendRightUncompressed((CompressedMatrixBlock) left, right, m, n);
			else
				return appendLeftUncompressed(left, (CompressedMatrixBlock) right, m, n);
		}
		if(isAligned((CompressedMatrixBlock) left, (CompressedMatrixBlock) right))
			return combineCompressed((CompressedMatrixBlock) left, (CompressedMatrixBlock) right);
		else
			return append((CompressedMatrixBlock) left, (CompressedMatrixBlock) right, m, n);
	}

	private static MatrixBlock cbindAllCompressed(CompressedMatrixBlock left, MatrixBlock[] right, int k)
		throws InterruptedException, ExecutionException {

		final int nCol = left.getNumColumns();
		for(int i = 0; i < right.length; i++) {
			CompressedMatrixBlock rightCM = ((CompressedMatrixBlock) right[i]);
			if(nCol != right[i].getNumColumns() || !isAligned(left, rightCM))
				return cbindAllNormalCompressed(left, right, k);
		}
		return cbindAllCompressedAligned(left, right, k);

	}

	private static boolean isAligned(CompressedMatrixBlock left, CompressedMatrixBlock right) {
		final List<AColGroup> gl = left.getColGroups();
		for(int j = 0; j < gl.size(); j++) {
			final AColGroup glj = gl.get(j);
			final int aColumnInGroup = glj.getColIndices().get(0);
			final AColGroup grj = right.getColGroupForColumn(aColumnInGroup);

			if(!glj.sameIndexStructure(grj) || glj.getNumCols() != grj.getNumCols())
				return false;

		}
		return true;
	}

	private static CompressedMatrixBlock combineCompressed(CompressedMatrixBlock left, CompressedMatrixBlock right) {
		final List<AColGroup> gl = left.getColGroups();
		final List<AColGroup> retCG = new ArrayList<>(gl.size());
		for(int j = 0; j < gl.size(); j++) {
			AColGroup glj = gl.get(j);
			int aColumnInGroup = glj.getColIndices().get(0);
			AColGroup grj = right.getColGroupForColumn(aColumnInGroup);
			// parallel combine...
			retCG.add(glj.combineWithSameIndex(left.getNumRows(), left.getNumColumns(), grj));
		}
		return new CompressedMatrixBlock(left.getNumRows(), left.getNumColumns() + right.getNumColumns(),
			left.getNonZeros() + right.getNonZeros(), false, retCG);
	}

	private static CompressedMatrixBlock cbindAllCompressedAligned(CompressedMatrixBlock left, MatrixBlock[] right,
		final int k) throws InterruptedException, ExecutionException {

		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final List<AColGroup> gl = left.getColGroups();
			final List<Future<AColGroup>> tasks = new ArrayList<>();
			final int nCol = left.getNumColumns();
			final int nRow = left.getNumRows();
			for(int i = 0; i < gl.size(); i++) {
				final AColGroup gli = gl.get(i);
				tasks.add(pool.submit(() -> {
					List<AColGroup> combines = new ArrayList<>();
					final int cId = gli.getColIndices().get(0);
					for(int j = 0; j < right.length; j++) {
						combines.add(((CompressedMatrixBlock) right[j]).getColGroupForColumn(cId));
					}
					return gli.combineWithSameIndex(nRow, nCol, combines);
				}));
			}

			final List<AColGroup> retCG = new ArrayList<>(gl.size());
			for(Future<AColGroup> t : tasks)
				retCG.add(t.get());

			int totalCol = nCol + right.length * nCol;

			return new CompressedMatrixBlock(left.getNumRows(), totalCol, -1, false, retCG);
		}
		finally {
			pool.shutdown();
		}

	}

	private static MatrixBlock appendLeftUncompressed(MatrixBlock left, CompressedMatrixBlock right, final int m,
		final int n) {

		final CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);

		final List<AColGroup> prev = right.getColGroups();
		final List<AColGroup> newGroup = new ArrayList<>(prev.size() + 1);
		final int nColL = left.getNumColumns();
		final IColIndex colIdx = ColIndexFactory.create(nColL);
		final AColGroup g = ColGroupUncompressed.create(colIdx, left, false);
		newGroup.add(g);
		for(AColGroup group : prev) {
			newGroup.add(group.shiftColIndices(nColL));
		}

		ret.allocateColGroupList(newGroup);
		ret.setNonZeros(left.getNonZeros() + right.getNonZeros());
		return ret;

	}

	private static MatrixBlock appendRightUncompressed(CompressedMatrixBlock left, MatrixBlock right, final int m,
		final int n) {

		final CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);
		final List<AColGroup> prev = left.getColGroups();
		final List<AColGroup> newGroup = new ArrayList<>(prev.size() + 1);
		newGroup.addAll(prev);
		final int cLenL = left.getNumColumns();
		final int cLenR = right.getNumColumns();
		final IColIndex colIdx = ColIndexFactory.create(cLenL, cLenR + cLenL);
		final AColGroup g = ColGroupUncompressed.create(colIdx, right, false);
		newGroup.add(g);
		ret.allocateColGroupList(newGroup);
		ret.setNonZeros(left.getNonZeros() + right.getNonZeros());
		return ret;
	}

	private static MatrixBlock append(CompressedMatrixBlock left, CompressedMatrixBlock right, final int m,
		final int n) {
		// init result matrix
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);
		appendColGroups(ret, left.getColGroups(), right.getColGroups(), left.getNumColumns());
		ret.setNonZeros(left.getNonZeros() + right.getNonZeros());
		ret.setOverlapping(left.isOverlapping() || right.isOverlapping());

		// final double compressedSize = ret.getInMemorySize();
		// final double uncompressedSize = MatrixBlock.estimateSizeInMemory(m, n, ret.getSparsity());

		// if(compressedSize < uncompressedSize)
		return ret;
		// else {
		// final double ratio = uncompressedSize / compressedSize;
		// String message = String.format("Decompressing c bind matrix because it had to small compression ratio: %2.3f",
		// ratio);
		// return ret.getUncompressed(message);
		// }
	}

	private static MatrixBlock appendRightEmpty(CompressedMatrixBlock left, MatrixBlock right, int m, int n) {
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);
		List<AColGroup> newGroup = new ArrayList<>(1);
		newGroup.add(ColGroupEmpty.create(right.getNumColumns()));
		appendColGroups(ret, left.getColGroups(), newGroup, left.getNumColumns());
		ret.setNonZeros(left.getNonZeros() + right.getNonZeros());
		ret.setOverlapping(left.isOverlapping());
		return ret;
	}

	private static MatrixBlock appendLeftEmpty(MatrixBlock left, CompressedMatrixBlock right, int m, int n) {
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);
		List<AColGroup> newGroup = new ArrayList<>(1);
		newGroup.add(ColGroupEmpty.create(left.getNumColumns()));
		appendColGroups(ret, newGroup, right.getColGroups(), left.getNumColumns());
		ret.setNonZeros(left.getNonZeros() + right.getNonZeros());
		ret.setOverlapping(right.isOverlapping());
		return ret;
	}

	private static void appendColGroups(CompressedMatrixBlock ret, List<AColGroup> left, List<AColGroup> right,
		int leftNumCols) {

		// shallow copy of lhs column groups
		ret.allocateColGroupList(new ArrayList<AColGroup>(left.size() + right.size()));

		for(AColGroup group : left)
			ret.getColGroups().add(group);

		for(AColGroup group : right)
			ret.getColGroups().add(group.shiftColIndices(leftNumCols));

		// meta data maintenance
		CLALibUtils.combineConstColumns(ret);
	}

	private static MatrixBlock uc(MatrixBlock mb) {
		// get uncompressed
		return CompressedMatrixBlock.getUncompressed(mb, "append");
	}
}
