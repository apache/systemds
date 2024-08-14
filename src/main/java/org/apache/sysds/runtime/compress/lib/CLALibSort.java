package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;

public class CLALibSort {

	public static MatrixBlock sort(CompressedMatrixBlock mb, MatrixValue weights, MatrixBlock result, int k) {
		// force uncompressed weights
		weights = CompressedMatrixBlock.getUncompressed(weights);

		if(mb.getNumColumns() == 1 && mb.getColGroups().size() == 1 && weights == null) {
			return sortSingleCol(mb, k);
		}

		// fallback to uncompressed.
		return CompressedMatrixBlock//
			.getUncompressed(mb, "sortOperations")//
			.sortOperations(weights, result);
	}

	private static MatrixBlock sortSingleCol(CompressedMatrixBlock mb, int k) {

		AColGroup g = mb.getColGroups().get(0);

		AColGroup r = g.sort();

		List<AColGroup> rg = new ArrayList<>();
		rg.add(r);
		return new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), false, rg);
	}
}
