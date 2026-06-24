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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;

public class CLALibRemoveEmpty {
	protected static final Log LOG = LogFactory.getLog(CLALibRemoveEmpty.class.getName());

	/**
	 * CP rmempty operation (single input, single output matrix)
	 * 
	 * @param in          The input matrix
	 * @param ret         The output matrix
	 * @param rows        If we are removing based on rows, or columns.
	 * @param emptyReturn Return row/column of zeros for empty input.
	 * @param select      An optional selection vector, to remove based on rather than empty rows or columns
	 * @return The result MatrixBlock, can be a different object that the caller used.
	 */
	public static MatrixBlock rmempty(CompressedMatrixBlock in, MatrixBlock ret, boolean rows, boolean emptyReturn,
		MatrixBlock select) {
		if(ret == null)
			ret = new MatrixBlock();
		MatrixBlock ret2 = LibMatrixReorg.rmemptyEarlyAbort(in, ret, rows, emptyReturn, select);
		if(ret2 != null)
			return ret2;

		if(rows)
			return rmEmptyRows(in, ret, emptyReturn, select);
		else
			return rmEmptyCols(in, ret, emptyReturn, select);
	}

	private static MatrixBlock rmEmptyCols(CompressedMatrixBlock in, MatrixBlock ret, boolean emptyReturn,
		MatrixBlock select) {
		if(select == null)
			return fallback(in, false, emptyReturn, select, ret);

		int cOut = (int) select.getNonZeros();
		if(cOut == -1)
			cOut = (int) select.recomputeNonZeros();
		if(cOut == 0){
			ret.reset(in.getNumRows(), !emptyReturn ? 0 : 1);
			return ret;
		}

		final boolean[] selectV = DataConverter
			.convertToBooleanVector(CompressedMatrixBlock.getUncompressed(select, "decompressing selection in rmempty"));

		final List<AColGroup> inG = in.getColGroups();
		final List<AColGroup> retG = new ArrayList<>(inG.size());
		try {
			for(int i = 0; i < inG.size(); i++) {
				AColGroup tmp = inG.get(i).removeEmptyCols(selectV);
				if(tmp != null)
					retG.add(tmp);
			}
		}
		catch(NotImplementedException e) {
			// Some column-group encodings (e.g. OLE/RLE) do not support index-only column removal;
			// decompress and remove on the uncompressed representation instead of failing.
			return fallback(in, false, emptyReturn, select, ret);
		}
		return new CompressedMatrixBlock(in.getNumRows(), cOut, -1, in.isOverlapping(), retG);

	}

	private static MatrixBlock rmEmptyRows(CompressedMatrixBlock in, MatrixBlock ret, boolean emptyReturn,
		MatrixBlock select) {
		if(select == null)
			return fallback(in, true, emptyReturn, select, ret);

		select = CompressedMatrixBlock.getUncompressed(select, "decompressing selection in rmempty");

		int rOut = (int) select.getNonZeros();
		if(rOut == -1)
			rOut = (int) select.recomputeNonZeros();
		if(rOut == 0){
			ret.reset(!emptyReturn ? 0 : 1, in.getNumColumns());
			return ret;
		}

		// TODO: add optimization to avoid linear scan and make selectV indexes, if selection is small relative to number
		// of rows
		// TODO: add decompress to boolean vector.
		final boolean[] selectV = DataConverter.convertToBooleanVector(select);



		final List<AColGroup> inG = in.getColGroups();
		final List<AColGroup> retG = new ArrayList<>(inG.size());
		try {
			for(int i = 0; i < inG.size(); i++) {
				retG.add(inG.get(i).removeEmptyRows(selectV, rOut));
			}
		}
		catch(NotImplementedException e) {
			// Some column-group encodings (e.g. OLE/RLE) do not support index-only row removal;
			// decompress and remove on the uncompressed representation instead of failing.
			return fallback(in, true, emptyReturn, select, ret);
		}

		return new CompressedMatrixBlock(rOut, in.getNumColumns(), -1, in.isOverlapping(), retG);
	}

	private static MatrixBlock fallback(CompressedMatrixBlock in, boolean rows, boolean emptyReturn, MatrixBlock select,
		MatrixBlock ret) {
		if(LOG.isDebugEnabled())
			LOG.debug("Decompressing for removeEmptyOperations with select: " + (select != null) + " rows: " + rows);
		MatrixBlock tmp = CompressedMatrixBlock.getUncompressed(in);
		MatrixBlock select2 = CompressedMatrixBlock.getUncompressed(select);
		return LibMatrixReorg.rmemptyUnsafe(tmp, ret, rows, emptyReturn, select2);
	}

}
