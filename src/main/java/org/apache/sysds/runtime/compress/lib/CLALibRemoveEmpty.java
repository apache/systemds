package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;

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

		return fallback(in, rows, emptyReturn, select, ret);
	}

	private static MatrixBlock rmEmptyRows(CompressedMatrixBlock in, MatrixBlock ret, boolean emptyReturn,
		MatrixBlock select) {
		if(select == null)
			return fallback(in, true, emptyReturn, select, ret);

		int rOut = (int) select.getNonZeros();
		if(rOut == -1)
			rOut = (int) select.recomputeNonZeros();
		
		//TODO: add optimization to avoid linear scan and make selectV indexes, if selection is small relative to number of rows
		//TODO: add decompress to boolean vector.
		final boolean[] selectV = DataConverter.convertToBooleanVector(CompressedMatrixBlock.getUncompressed(select, "decompressing selection in rmempty"));

		final List<AColGroup> inG = in.getColGroups();
		final List<AColGroup> retG = new ArrayList<>(inG.size());
		for(int i = 0; i < inG.size(); i++) {
			retG.add(inG.get(i).removeEmptyRows(selectV, rOut));
		}

		return new CompressedMatrixBlock(rOut, in.getNumColumns(), -1, in.isOverlapping(), retG);
	}

	private static MatrixBlock fallback(CompressedMatrixBlock in, boolean rows, boolean emptyReturn, MatrixBlock select,
		MatrixBlock ret) {
		LOG.warn("Decompressing because: removeEmptyOperations  with select: " + (select != null) + " rows: " + rows);
		MatrixBlock tmp = CompressedMatrixBlock.getUncompressed(in);
		MatrixBlock select2 = CompressedMatrixBlock.getUncompressed(select);
		return LibMatrixReorg.rmemptyUnsafe(tmp, ret, rows, emptyReturn, select2);
	}

}
