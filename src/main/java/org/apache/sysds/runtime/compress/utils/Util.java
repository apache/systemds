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

package org.apache.sysds.runtime.compress.utils;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public interface Util {

	static final Log LOG = LogFactory.getLog(Util.class.getName());

	public static int[] combine(int[] lhs, int[] rhs) {
		int[] joined = new int[lhs.length + rhs.length];
		int lp = 0;
		int rp = 0;
		int i = 0;
		for(; i < joined.length && lp < lhs.length && rp < rhs.length; i++) {
			if(lhs[lp] < rhs[rp])
				joined[i] = lhs[lp++];
			else
				joined[i] = rhs[rp++];
		}

		while(lp < lhs.length)
			joined[i++] = lhs[lp++];

		while(rp < rhs.length)
			joined[i++] = rhs[rp++];

		return joined;
	}

	public static int getPow2(int x) {
		int v = UtilFunctions.nextIntPow2(x + 1);
		return Math.max(v, 4);
	}

	public static int[] genColsIndices(final int numCols) {
		final int[] colIndices = new int[numCols];
		for(int i = 0; i < numCols; i++)
			colIndices[i] = i;
		return colIndices;
	}

	public static int[] genColsIndices(final int cl, final int cu) {
		final int[] colIndices = new int[cu - cl];
		for(int i = 0, j = cl; j < cu; i++, j++)
			colIndices[i] = j;
		return colIndices;
	}

	public static int[] genColsIndicesOffset(final int numCols, final int start) {
		final int[] colIndices = new int[numCols];
		for(int i = 0, j = start; i < numCols; i++, j++)
			colIndices[i] = j;
		return colIndices;
	}

	public static MatrixBlock matrixBlockFromDenseArray(double[] values, int nCol) {
		return matrixBlockFromDenseArray(values, nCol, true);
	}

	public static MatrixBlock matrixBlockFromDenseArray(double[] values, int nCol, boolean check) {
		final int nRow = values.length / nCol;
		DenseBlock dictV = new DenseBlockFP64(new int[] {nRow, nCol}, values);
		MatrixBlock ret = new MatrixBlock(nRow, nCol, dictV);
		if(check) {
			ret.recomputeNonZeros();
			ret.examSparsity(true);
		}
		else
			ret.setNonZeros(-1);

		return ret;
	}

	public static MatrixBlock extractValues(double[] v, int[] colIndexes) {
		MatrixBlock rowVector = new MatrixBlock(1, colIndexes.length, false);
		for(int i = 0; i < colIndexes.length; i++)
			rowVector.quickSetValue(0, i, v[colIndexes[i]]);
		return rowVector;
	}
}
