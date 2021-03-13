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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CLALibReExpand {

	// private static final Log LOG = LogFactory.getLog(CLALibReExpand.class.getName());

	public static MatrixBlock reExpand(CompressedMatrixBlock in, MatrixBlock ret, double max, boolean cast,
			boolean ignore, int k) {
		int iMax = UtilFunctions.toInt(max);
		if (!ignore && in.min() == 0.0)
			throw new DMLRuntimeException("Invalid input w/ zeros for rexpand ignore=false " + "(rlen="
					+ in.getNumRows() + ", nnz=" + in.getNonZeros() + ").");

		if (in.isOverlapping() || in.getColGroups().size() > 1) {
			throw new DMLRuntimeException(
					"Invalid input for re expand operations, currently not supporting overlapping or multi column groups");
		}

		// check for empty inputs (for ignore=true)
		if (in.isEmptyBlock(false)) {
			ret.reset(in.getNumRows(), iMax, true);
			return ret;
		}
		CompressedMatrixBlock retC = ret instanceof CompressedMatrixBlock ? (CompressedMatrixBlock) ret
				: new CompressedMatrixBlock(in.getNumRows(), iMax);

		return reExpandRows(in, retC, iMax, cast, k);
	}

	private static CompressedMatrixBlock reExpandRows(CompressedMatrixBlock in, CompressedMatrixBlock ret, int max,
			boolean cast, int k) {
		ColGroupValue oldGroup = ((ColGroupValue) in.getColGroups().get(0));

		ADictionary newDictionary = oldGroup.getDictionary().reExpandColumns(max);
		AColGroup newGroup = oldGroup.copyAndSet(getColIndexes(max), newDictionary);
		List<AColGroup> newColGroups = new ArrayList<>(1);
		newColGroups.add(newGroup);

		ret.allocateColGroupList(newColGroups);
		ret.setOverlapping(true);

		ret.recomputeNonZeros();
		return ret;
	}

	private static int[] getColIndexes(int max) {
		int[] ret = new int[max];
		for (int i = 0; i < max; i++) {
			ret[i] = i;
		}
		return ret;
	}
}
