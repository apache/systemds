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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CLALibAppend {

	private static final Log LOG = LogFactory.getLog(CLALibAppend.class.getName());

	public static MatrixBlock append(MatrixBlock left, MatrixBlock right) {
		
		if(left.isEmpty())
			return right;
		else if(right.isEmpty())
			return left;
		final int m = left.getNumRows();
		final int n = left.getNumColumns() + right.getNumColumns();
		long nnz = left.getNonZeros() + right.getNonZeros();
		if(left.getNonZeros() < 0 || right.getNonZeros() < 0)
			nnz = -1;

		// try to compress both sides (if not already compressed).
		if(!(left instanceof CompressedMatrixBlock) && m > 1000){
			LOG.info("Compressing left for append operation");
			Pair<MatrixBlock, CompressionStatistics> x = CompressedMatrixBlockFactory.compress(left);
			if(x.getRight().ratio > 3.0)
				left = x.getLeft();
			
		}
		if(!(right instanceof CompressedMatrixBlock) && m > 1000){
			LOG.info("Compressing right for append operation");
			Pair<MatrixBlock, CompressionStatistics> x = CompressedMatrixBlockFactory.compress(right);
			if(x.getRight().ratio > 3.0)
				right = x.getLeft();
		}

		// if compression failed then use default append method.
		if(!(left instanceof CompressedMatrixBlock && right instanceof CompressedMatrixBlock))
			return uc(left).append(uc(right), new MatrixBlock());

		CompressedMatrixBlock leftC = (CompressedMatrixBlock) left;
		CompressedMatrixBlock rightC = (CompressedMatrixBlock) right;

		// init result matrix
		CompressedMatrixBlock ret = new CompressedMatrixBlock(m, n);

		// shallow copy of lhs column groups
		ret.allocateColGroupList(new ArrayList<AColGroup>());

		for(AColGroup group : leftC.getColGroups()){
			AColGroup tmp = group.copy();
			ret.getColGroups().add(tmp);
		}
		for(AColGroup group : rightC.getColGroups()) {
			AColGroup tmp = group.copy();
			tmp.shiftColIndices(left.getNumColumns());
			ret.getColGroups().add(tmp);
		}

		// meta data maintenance
		ret.setNonZeros(nnz);
		LOG.error(ret);
		return ret;
	}

	private static MatrixBlock uc(MatrixBlock mb) {
		// get uncompressed
		return CompressedMatrixBlock.getUncompressed(mb);
	}

}
