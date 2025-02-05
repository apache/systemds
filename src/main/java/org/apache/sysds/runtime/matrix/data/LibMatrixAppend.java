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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.lib.CLALibCBind;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;

public class LibMatrixAppend {

	public static MatrixBlock append(MatrixBlock a, MatrixBlock[] that, MatrixBlock result, boolean cbind) {
		int rlen = a.rlen;
		int clen = a.clen;
		long nonZeros = a.nonZeros;

		checkDimensionsForAppend(that, cbind, rlen, clen);

		for(int k = 0; k < that.length; k++)
			if(that[k] instanceof CompressedMatrixBlock) {
				if(that.length == 1 && cbind)
					return CLALibCBind.cbind(a, that[0], 1);
				that[k] = CompressedMatrixBlock.getUncompressed(that[k], "Append N");
			}

		final int m = cbind ? rlen : combinedRows(that, rlen);
		final int n = cbind ? combinedCols(that, clen) : clen;
		final long nnz = calculateCombinedNNz(that, nonZeros);

		boolean shallowCopy = (nonZeros == nnz);
		boolean sp = MatrixBlock.evalSparseFormatInMemory(m, n, nnz);

		// init result matrix
		if(result == null)
			result = new MatrixBlock(m, n, sp, nnz);
		else
			result.reset(m, n, sp, nnz);

		// core append operation
		// copy left and right input into output
		if(!result.sparse && nnz != 0)
			appendDense(a, that, result, cbind, rlen, m, n);
		else if(nnz != 0)
			appendSparse(a, that, result, cbind, rlen, clen, nnz, shallowCopy);

		// update meta data
		result.nonZeros = nnz;
		return result;
	}

	private static void appendSparse(MatrixBlock a, MatrixBlock[] that, MatrixBlock result, boolean cbind, int rlen,
		int clen, final long nnz, boolean shallowCopy) {
		// adjust sparse rows if required
		result.allocateSparseRowsBlock();
		// allocate sparse rows once for cbind
		if(cbind && nnz > rlen && !shallowCopy && result.getSparseBlock() instanceof SparseBlockMCSR) {
			final SparseBlock sblock = result.getSparseBlock();
			// for each row calculate how many non zeros are pressent.
			for(int i = 0; i < result.rlen; i++)
				sblock.allocate(i, computeNNzRow(that, i, a));

		}

		// core append operation
		// we can always append this directly to offset 0.0 in both cbind and rbind.
		result.appendToSparse(a, 0, 0, !shallowCopy);
		if(cbind) {
			for(int i = 0, off = clen; i < that.length; i++) {
				result.appendToSparse(that[i], 0, off);
				off += that[i].clen;
			}
		}
		else { // rbind
			for(int i = 0, off = rlen; i < that.length; i++) {
				result.appendToSparse(that[i], off, 0);
				off += that[i].rlen;
			}
		}
	}

	private static void appendDense(MatrixBlock a, MatrixBlock[] that, MatrixBlock result, boolean cbind, int rlen,
		final int m, final int n) {
		if(cbind) {
			DenseBlock resd = result.allocateBlock().getDenseBlock();
			MatrixBlock[] in = ArrayUtils.addAll(new MatrixBlock[] {a}, that);

			for(int i = 0; i < m; i++) {
				for(int k = 0, off = 0; k < in.length; off += in[k].clen, k++) {
					if(in[k].isEmptyBlock(false))
						continue;
					if(in[k].sparse) {
						SparseBlock src = in[k].sparseBlock;
						if(src.isEmpty(i))
							continue;
						int srcpos = src.pos(i);
						int srclen = src.size(i);
						int[] srcix = src.indexes(i);
						double[] srcval = src.values(i);
						double[] resval = resd.values(i);
						int resix = resd.pos(i, off);
						for(int j = srcpos; j < srcpos + srclen; j++)
							resval[resix + srcix[j]] = srcval[j];
					}
					else {
						DenseBlock src = in[k].getDenseBlock();
						double[] srcval = src.values(i);
						double[] resval = resd.values(i);
						System.arraycopy(srcval, src.pos(i), resval, resd.pos(i, off), in[k].clen);
					}
				}
			}
		}
		else { // rbind
			result.copy(0, rlen - 1, 0, n - 1, a, false);
			for(int i = 0, off = rlen; i < that.length; i++) {
				result.copy(off, off + that[i].rlen - 1, 0, n - 1, that[i], false);
				off += that[i].rlen;
			}
		}
	}

	public static void checkDimensionsForAppend(MatrixBlock[] in, boolean cbind, int rlen, int clen) {
		if(cbind) {
			for(int i = 0; i < in.length; i++)
				if(in[i].rlen != rlen)
					throw new DMLRuntimeException(
						"Invalid nRow dimension for append cbind: was " + in[i].rlen + " should be: " + rlen);
		}
		else {
			for(int i = 0; i < in.length; i++)
				if(in[i].clen != clen)
					throw new DMLRuntimeException(
						"Invalid nCol dimension for append rbind: was " + in[i].clen + " should be: " + clen);
		}
	}

	private static int combinedRows(MatrixBlock[] that, int rlen) {
		int r = rlen;
		for(MatrixBlock b : that)
			r += b.rlen;
		return r;
	}

	private static int combinedCols(MatrixBlock[] that, int clen) {
		int c = clen;
		for(MatrixBlock b : that)
			c += b.clen;
		return c;
	}

	private static long calculateCombinedNNz(MatrixBlock[] that, long nonZeros) {
		long nnz = nonZeros;
		for(MatrixBlock b : that)
			nnz += b.nonZeros;
		return nnz;
	}

	private static int computeNNzRow(MatrixBlock[] that, int row, MatrixBlock a) {
		int lnnz = (int) a.recomputeNonZeros(row, row, 0, a.clen - 1);
		for(MatrixBlock b : that)
			lnnz += b.recomputeNonZeros(row, row, 0, b.clen - 1);
		return lnnz;
	}
}
