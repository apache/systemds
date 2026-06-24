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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public final class CLALibSort {

	private CLALibSort() {
		// private constructor for utility class.
	}

	/**
	 * Sort (order) a compressed matrix in place of the {@code order} built-in, while keeping the result compressed.
	 *
	 * The compressed fast-path only supports the case the user can benefit from: a single column held in a single column
	 * group, sorted ascending and returning the sorted values (not the index permutation). For everything else (multiple
	 * columns, multiple column groups, descending order, index return, or a column-group encoding without a sort
	 * implementation) this returns {@code null} so the caller can fall back to a decompressed reorg.
	 *
	 * @param mb the compressed matrix to sort
	 * @param fn the sort specification carried by the reorg operator
	 * @return the sorted compressed matrix, or {@code null} if the compressed fast-path does not apply
	 */
	public static MatrixBlock sort(CompressedMatrixBlock mb, SortIndex fn) {
		final boolean singleColumn = mb.getNumColumns() == 1 && mb.getColGroups().size() == 1;
		if(!singleColumn || fn.getDecreasing() || fn.getIndexReturn())
			return null;

		try {
			final AColGroup g = mb.getColGroups().get(0);
			final AColGroup sorted = g.sort();
			final List<AColGroup> rg = new ArrayList<>(1);
			rg.add(sorted);
			return new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), false, rg);
		}
		catch(NotImplementedException e) {
			// the column-group encoding does not implement sort -> let the caller decompress.
			return null;
		}
	}
}
