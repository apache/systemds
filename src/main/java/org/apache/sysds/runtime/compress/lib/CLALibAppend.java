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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CLALibAppend {

	private static final Log LOG = LogFactory.getLog(CLALibAppend.class.getName());

	public static MatrixBlock append(MatrixBlock left, MatrixBlock right, int k) {

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
			final double spar = (double) (left.getNonZeros() + right.getNonZeros()) / ((double) m * n);
			final double estSizeUncompressed = MatrixBlock.estimateSizeInMemory(m, n, spar);
			final double estSizeCompressed = left.getInMemorySize() + right.getInMemorySize();
			if(estSizeUncompressed < estSizeCompressed)
				return uc(left).append(uc(right), null);
			else if(left instanceof CompressedMatrixBlock)
				return appendRightUncompressed((CompressedMatrixBlock) left, right, m, n);
			else
				return appendLeftUncompressed(left, (CompressedMatrixBlock) right, m, n);
		}

		return append((CompressedMatrixBlock) left, (CompressedMatrixBlock) right, m, n);
	}

	private static MatrixBlock appendLeftUncompressed(MatrixBlock left, CompressedMatrixBlock right, final int m,
		final int n) {

		final CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);

		final List<AColGroup> prev = right.getColGroups();
		final List<AColGroup> newGroup = new ArrayList<>(prev.size() + 1);
		final int nColL = left.getNumColumns();
		final int[] colIdx = Util.genColsIndices(nColL);
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
		final int[] colIdx = Util.genColsIndicesOffset(right.getNumColumns(), left.getNumColumns());
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

		final double compressedSize = ret.getInMemorySize();
		final double uncompressedSize = MatrixBlock.estimateSizeInMemory(m, n, ret.getSparsity());

		if(compressedSize < uncompressedSize)
			return ret;
		else {
			final double ratio = uncompressedSize / compressedSize;
			String message = String.format("Decompressing c bind matrix because it had to small compression ratio: %2.3f",
				ratio);
			return ret.getUncompressed(message);
		}
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
