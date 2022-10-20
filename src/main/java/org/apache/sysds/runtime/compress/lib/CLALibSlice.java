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
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CLALibSlice {

	protected static final Log LOG = LogFactory.getLog(CLALibSlice.class.getName());

	/**
	 * Slice blocks of compressed matrices from the compressed representation.
	 * 
	 * @param cmb  The input block to slice.
	 * @param blen The length of the blocks.
	 * @return A list containing CompressedMatrixBlocks or MatrixBlocks
	 */
	public static List<MatrixBlock> sliceBlocks(CompressedMatrixBlock cmb, int blen) {
		final List<MatrixBlock> mbs = new ArrayList<>();
		for(int b = 0; b < cmb.getNumRows(); b += blen)
			mbs.add(sliceRowsCompressed(cmb, b, Math.min(b + blen, cmb.getNumRows()) - 1));
		return mbs;
	}

	public static MatrixBlock slice(CompressedMatrixBlock cmb, int rl, int ru, int cl, int cu, boolean deep) {
		if(rl == ru && cl == cu)
			return sliceSingle(cmb, rl, cl);
		else if(rl == 0 && ru == cmb.getNumRows() - 1)
			return sliceColumns(cmb, cl, cu);
		else if(cl == 0 && cu == cmb.getNumColumns() - 1)
			return sliceRows(cmb, rl, ru, deep);
		else
			return sliceInternal(cmb, rl, ru, cl, cu, deep);
	}

	private static MatrixBlock sliceInternal(CompressedMatrixBlock cmb, int rl, int ru, int cl, int cu, boolean deep) {
		/**
		 * In the case where an internal matrix is sliced out, then first slice out the columns to an compressed
		 * intermediate. Then call slice recursively, to do the row slice. Since we do not copy the index structure but
		 * simply maintain a pointer to the original this is fine.
		 */
		return sliceRows(sliceColumns(cmb, cl, cu), rl, ru, deep);
	}

	private static MatrixBlock sliceRows(CompressedMatrixBlock cmb, int rl, int ru, boolean deep) {
		if(shouldDecompressSliceRows(cmb, rl, ru))
			return sliceRowsDecompress(cmb, rl, ru);
		else
			return sliceRowsCompressed(cmb, rl, ru);
	}

	private static boolean shouldDecompressSliceRows(CompressedMatrixBlock cmb, int rl, int ru) {
		return ru - rl > 124;
	}

	private static MatrixBlock sliceRowsDecompress(CompressedMatrixBlock cmb, int rl, int ru) {
		final int nCol = cmb.getNumColumns();
		final int rue = ru + 1;
		MatrixBlock tmp = new MatrixBlock(rue - rl, nCol, false).allocateDenseBlock();
		DenseBlock db = tmp.getDenseBlock();
		final List<AColGroup> groups = cmb.getColGroups();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			final double[] constV = new double[nCol];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);
			for(AColGroup g : filteredGroups)
				g.decompressToDenseBlock(db, rl, rue, -rl, 0);
			AColGroup cRet = ColGroupConst.create(constV);
			cRet.decompressToDenseBlock(db, rl, rue, -rl, 0);
		}
		else
			for(AColGroup g : groups)
				g.decompressToDenseBlock(db, rl, rue, -rl, 0);

		tmp.recomputeNonZeros();
		tmp.examSparsity();
		return tmp;
	}

	private static MatrixBlock sliceRowsCompressed(CompressedMatrixBlock cmb, int rl, int ru) {
		final List<AColGroup> groups = cmb.getColGroups();
		final List<AColGroup> newColGroups = new ArrayList<AColGroup>(groups.size());
		final int rue = ru + 1;

		final CompressedMatrixBlock ret = new CompressedMatrixBlock(rue - rl, cmb.getNumColumns());

		for(AColGroup grp : cmb.getColGroups()) {
			final AColGroup slice = grp.sliceRows(rl, rue);
			if(slice != null)
				newColGroups.add(slice);
		}

		if(newColGroups.size() == 0)
			return new MatrixBlock(rue - rl, cmb.getNumColumns(), 0.0);

		ret.allocateColGroupList(newColGroups);
		ret.recomputeNonZeros();
		ret.setOverlapping(cmb.isOverlapping());
		return ret;
	}

	private static MatrixBlock sliceSingle(CompressedMatrixBlock cmb, int row, int col) {
		// get a single index, and return in a matrixBlock
		MatrixBlock tmp = new MatrixBlock(1, 1, 0);
		tmp.setValue(0, 0, cmb.getValue(row, col));
		return tmp;
	}

	public static CompressedMatrixBlock sliceColumns(CompressedMatrixBlock cmb, int cl, int cu) {
		final int cue = cu + 1;
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(cmb.getNumRows(), cue - cl);

		final List<AColGroup> newColGroups = new ArrayList<>();
		for(AColGroup grp : cmb.getColGroups()) {
			final AColGroup slice = grp.sliceColumns(cl, cue);
			if(slice != null)
				newColGroups.add(slice);
		}

		ret.allocateColGroupList(newColGroups);
		ret.recomputeNonZeros();
		ret.setOverlapping(cmb.isOverlapping());
		return ret;
	}
}
