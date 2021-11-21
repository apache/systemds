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

		if(!(left instanceof CompressedMatrixBlock)) {
			LOG.info("Trying to compress left side of append");
			left = CompressedMatrixBlockFactory.compress(left, k).getLeft();
		}

		if(!(right instanceof CompressedMatrixBlock)) {
			LOG.info("Trying to compress right side of append");
			right = CompressedMatrixBlockFactory.compress(right, k).getLeft();
		}

		// if compression failed then use default append method.
		if(!(left instanceof CompressedMatrixBlock && right instanceof CompressedMatrixBlock))
			return uc(left).append(uc(right), null);

		return append((CompressedMatrixBlock) left, (CompressedMatrixBlock) right, true);

	}

	public static MatrixBlock append(CompressedMatrixBlock left, CompressedMatrixBlock right, boolean check) {
		final int m = left.getNumRows();
		final int n = left.getNumColumns() + right.getNumColumns();
		// init result matrix
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);

		ret = appendColGroups(ret, left.getColGroups(), right.getColGroups(), left.getNumColumns());

		ret.setOverlapping(left.isOverlapping() || right.isOverlapping());
		double compressedSize = ret.getInMemorySize();
		double uncompressedSize = MatrixBlock.estimateSizeInMemory(m, n, ret.getSparsity());

		double ratio = uncompressedSize / compressedSize;

		if(!check || compressedSize < uncompressedSize)
			return ret;
		else {
			String message = String.format("Decompressing c bind matrix because it had to small compression ratio: %2.3f",
				ratio);
			return ret.getUncompressed(message);
		}
	}

	private static MatrixBlock appendRightEmpty(CompressedMatrixBlock left, MatrixBlock right, int m, int n) {
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);
		List<AColGroup> newGroup = new ArrayList<>(1);
		newGroup.add(ColGroupEmpty.generate(right.getNumColumns()));
		ret = appendColGroups(ret, left.getColGroups(), newGroup, left.getNumColumns());
		ret.setOverlapping(left.isOverlapping());
		return ret;
	}

	private static MatrixBlock appendLeftEmpty(MatrixBlock left, CompressedMatrixBlock right, int m, int n) {
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);
		List<AColGroup> newGroup = new ArrayList<>(1);
		newGroup.add(ColGroupEmpty.generate(left.getNumColumns()));
		ret = appendColGroups(ret, newGroup, right.getColGroups(), left.getNumColumns());
		ret.setOverlapping(right.isOverlapping());
		return ret;
	}

	private static CompressedMatrixBlock appendColGroups(CompressedMatrixBlock ret, List<AColGroup> left,
		List<AColGroup> right, int leftNumCols) {

		// shallow copy of lhs column groups
		ret.allocateColGroupList(new ArrayList<AColGroup>(left.size() + right.size()));

		final int nRows = ret.getNumRows();
		long nnz = 0;
		for(AColGroup group : left) {
			AColGroup tmp = group.copy();
			ret.getColGroups().add(tmp);
			nnz += group.getNumberNonZeros(nRows);
		}

		for(AColGroup group : right) {
			AColGroup tmp = group.copy();
			tmp.shiftColIndices(leftNumCols);
			ret.getColGroups().add(tmp);
			nnz += group.getNumberNonZeros(nRows);
		}

		// meta data maintenance
		ret.setNonZeros(nnz);
		CLALibUtils.combineConstColumns(ret);
		return ret;
	}

	private static MatrixBlock uc(MatrixBlock mb) {
		// get uncompressed
		return CompressedMatrixBlock.getUncompressed(mb);
	}

}
