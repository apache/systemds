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

import java.util.List;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class CLALibMMChain {

	public static MatrixBlock mmChain(CompressedMatrixBlock x, MatrixBlock v, MatrixBlock w, MatrixBlock out,
		ChainType ctype, int k) {

		if(x.isEmpty())
			return returnEmpty(x, out);

		// Morph the columns to effecient types for the operation.
		x = filterColGroups(x);

		// Allow overlapping intermediate if the intermediate is guaranteed not to be overlapping.
		final boolean allowOverlap = x.getColGroups().size() == 1 && isOverlappingAllowed();

		// Right hand side multiplication
		MatrixBlock tmp = CLALibRightMultBy.rightMultByMatrix(x, v, null, k, allowOverlap);

		if(ctype == ChainType.XtwXv) // Multiply intermediate with vector if needed
			tmp = binaryMultW(tmp, w, k);

		if(tmp instanceof CompressedMatrixBlock)
			// Compressed Compressed Matrix Multiplication
			CLALibLeftMultBy.leftMultByMatrixTransposed(x, (CompressedMatrixBlock) tmp, out, k);
		else
			// LMM with Compressed - uncompressed multiplication.
			CLALibLeftMultBy.leftMultByMatrixTransposed(x, tmp, out, k);

		if(out.getNumColumns() != 1) // transpose the output to make it a row output if needed
			out = LibMatrixReorg.transposeInPlace(out, k);

		return out;
	}

	private static boolean isOverlappingAllowed() {
		return ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.COMPRESSED_OVERLAPPING);
	}

	private static MatrixBlock returnEmpty(CompressedMatrixBlock x, MatrixBlock out) {
		out = prepareReturn(x, out);
		return out;
	}

	private static MatrixBlock prepareReturn(CompressedMatrixBlock x, MatrixBlock out) {
		final int clen = x.getNumColumns();
		if(out != null)
			out.reset(clen, 1, false);
		else
			out = new MatrixBlock(clen, 1, false);
		return out;
	}

	private static MatrixBlock binaryMultW(MatrixBlock tmp, MatrixBlock w, int k) {
		final BinaryOperator bop = new BinaryOperator(Multiply.getMultiplyFnObject(), k);
		if(tmp instanceof CompressedMatrixBlock)
			tmp = CLALibBinaryCellOp.binaryOperationsRight(bop, (CompressedMatrixBlock) tmp, w, null);
		else
			LibMatrixBincell.bincellOpInPlace(tmp, w, bop);
		return tmp;
	}

	private static CompressedMatrixBlock filterColGroups(CompressedMatrixBlock x) {
		final List<AColGroup> groups = x.getColGroups();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			final int nCol = x.getNumColumns();
			final double[] constV = new double[nCol];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);

			AColGroup c = ColGroupFactory.genColGroupConst(constV);
			filteredGroups.add(c);
			x.allocateColGroupList(filteredGroups);
			return x;
		}
		else
			return x;
	}
}
