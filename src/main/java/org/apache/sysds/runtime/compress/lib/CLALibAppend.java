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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CLALibAppend {

	private static final Log LOG = LogFactory.getLog(CLALibAppend.class.getName());

	public static MatrixBlock append(MatrixBlock left, MatrixBlock right) {

		if(left.isEmpty() && right instanceof CompressedMatrixBlock)
			return appendLeftEmpty(left, (CompressedMatrixBlock) right);
		else if(right.isEmpty() && left instanceof CompressedMatrixBlock)
			return appendRightEmpty((CompressedMatrixBlock)left, right);

		final int m = left.getNumRows();
		final int n = left.getNumColumns() + right.getNumColumns();

		// try to compress both sides (if not already compressed).
		if(!(left instanceof CompressedMatrixBlock) && m > 1000) {
			LOG.warn("Compressing left for append operation");
			Pair<MatrixBlock, CompressionStatistics> x = CompressedMatrixBlockFactory.compress(left);
			if(x.getRight().ratio > 3.0)
				left = x.getLeft();

		}
		if(!(right instanceof CompressedMatrixBlock) && m > 1000) {
			LOG.warn("Compressing right for append operation");
			Pair<MatrixBlock, CompressionStatistics> x = CompressedMatrixBlockFactory.compress(right);
			if(x.getRight().ratio > 3.0)
				right = x.getLeft();
		}

		// if compression failed then use default append method.
		if(!(left instanceof CompressedMatrixBlock && right instanceof CompressedMatrixBlock))
			return uc(left).append(uc(right), null);

		CompressedMatrixBlock leftC = (CompressedMatrixBlock) left;
		CompressedMatrixBlock rightC = (CompressedMatrixBlock) right;

		// init result matrix
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);

		ret = appendColGroups(ret, leftC.getColGroups(), rightC.getColGroups(), leftC.getNumColumns());
		return ret;
	}

	private static MatrixBlock appendRightEmpty(CompressedMatrixBlock left, MatrixBlock right) {

		final int m = left.getNumRows();
		final int n = left.getNumColumns() + right.getNumColumns();
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m,n);

		List<AColGroup> newGroup = new ArrayList<>(1);
		newGroup.add(ColGroupEmpty.generate(right.getNumColumns(), right.getNumRows()));
		ret = appendColGroups(ret, left.getColGroups(), newGroup, left.getNumColumns());

		return ret;
	}

	private static MatrixBlock appendLeftEmpty(MatrixBlock left, CompressedMatrixBlock right) {
		final int m = left.getNumRows();
		final int n = left.getNumColumns() + right.getNumColumns();
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m,n);

		List<AColGroup> newGroup = new ArrayList<>(1);
		newGroup.add(ColGroupEmpty.generate(left.getNumColumns(), left.getNumRows()));
		ret = appendColGroups(ret, newGroup, right.getColGroups(), left.getNumColumns());

		return ret;
	}

	private static CompressedMatrixBlock appendColGroups(CompressedMatrixBlock ret, List<AColGroup> left,
		List<AColGroup> right, int leftNumCols) {

		// shallow copy of lhs column groups
		ret.allocateColGroupList(new ArrayList<AColGroup>(left.size() + right.size()));

		long nnz = 0;
		for(AColGroup group : left) {
			AColGroup tmp = group.copy();
			ret.getColGroups().add(tmp);
			nnz += group.getNumberNonZeros();
		}

		for(AColGroup group : right) {
			AColGroup tmp = group.copy();
			tmp.shiftColIndices(leftNumCols);
			ret.getColGroups().add(tmp);
			nnz += group.getNumberNonZeros();
		}

		// meta data maintenance
		ret.setNonZeros(nnz);
		return ret;
	}

	private static MatrixBlock uc(MatrixBlock mb) {
		// get uncompressed
		return CompressedMatrixBlock.getUncompressed(mb);
	}

}
