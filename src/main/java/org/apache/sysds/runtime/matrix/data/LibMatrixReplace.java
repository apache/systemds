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

import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;

public class LibMatrixReplace {

	private LibMatrixReplace() {

	}

	public static MatrixBlock replaceOperations(MatrixBlock in, MatrixBlock ret, double pattern, double replacement) {
		return replaceOperations(in, ret, pattern, replacement, InfrastructureAnalyzer.getLocalParallelism());
	}

	public static MatrixBlock replaceOperations(MatrixBlock in, MatrixBlock ret, double pattern, double replacement,
		int k) {

		// ensure input its in the right format
		in.examSparsity(k);

		final int rlen = in.getNumRows();
		final int clen = in.getNumColumns();
		final long nonZeros = in.getNonZeros();
		final boolean sparse = in.isInSparseFormat();

		if(ret != null)
			ret.reset(rlen, clen, sparse);
		else
			ret = new MatrixBlock(rlen, clen, sparse);

		// probe early abort conditions
		if(nonZeros == 0 && pattern != 0)
			return ret;
		if(!in.containsValue(pattern))
			return in; // avoid allocation + copy
		if(in.isEmpty() && pattern == 0) {
			ret.reset(rlen, clen, replacement);
			return ret;
		}

		final boolean replaceNaN = Double.isNaN(pattern);

		final long nnz;
		if(sparse) // SPARSE
			nnz = replaceSparse(in, ret, pattern, replacement, replaceNaN);
		else if(replaceNaN)
			nnz = replaceDenseNaN(in, ret, replacement);
		else
			nnz = replaceDense(in, ret, pattern, replacement);

		ret.setNonZeros(nnz);
		ret.examSparsity(k);
		return ret;
	}

	private static long replaceSparse(MatrixBlock in, MatrixBlock ret, double pattern, double replacement,
		boolean replaceNaN) {
		if(replaceNaN)
			return replaceSparseInSparseOutReplaceNaN(in, ret, replacement);
		else if(pattern != 0d) // sparse safe.
			return replaceSparseInSparseOut(in, ret, pattern, replacement);
		else // sparse unsafe
			return replace0InSparse(in, ret, replacement);

	}

	private static long replaceSparseInSparseOutReplaceNaN(MatrixBlock in, MatrixBlock ret, double replacement) {
		ret.allocateSparseRowsBlock();
		SparseBlock a = in.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		long nnz = 0;
		for(int i = 0; i < in.rlen; i++) {
			if(!a.isEmpty(i)) {
				int apos = a.pos(i);
				int alen = a.size(i);
				c.allocate(i, alen);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				for(int j = apos; j < apos + alen; j++) {
					double val = avals[j];
					if(Double.isNaN(val))
						c.append(i, aix[j], replacement);
					else
						c.append(i, aix[j], val);
				}
				c.compact(i);
				nnz += c.size(i);
			}
		}
		return nnz;
	}

	private static long replaceSparseInSparseOut(MatrixBlock in, MatrixBlock ret, double pattern, double replacement) {
		ret.allocateSparseRowsBlock();
		final SparseBlock a = in.sparseBlock;
		final SparseBlock c = ret.sparseBlock;

		return replaceSparseInSparseOut(a, c, pattern, replacement, 0, in.rlen);

	}

	private static long replaceSparseInSparseOut(SparseBlock a, SparseBlock c, double pattern, double replacement, int s,
		int e) {
		long nnz = 0;
		for(int i = s; i < e; i++) {
			if(!a.isEmpty(i)) {
				final int apos = a.pos(i);
				final int alen = a.size(i);
				final int[] aix = a.indexes(i);
				final double[] avals = a.values(i);
				c.allocate(i, alen);
				for(int j = apos; j < apos + alen; j++) {
					double val = avals[j];
					if(val == pattern)
						c.append(i, aix[j], replacement);
					else
						c.append(i, aix[j], val);
				}
				c.compact(i);
				nnz += c.size(i);
			}
		}
		return nnz;
	}

	private static long replace0InSparse(MatrixBlock in, MatrixBlock ret, double replacement) {
		ret.sparse = false;
		ret.allocateDenseBlock();
		SparseBlock a = in.sparseBlock;
		DenseBlock c = ret.getDenseBlock();

		// initialize with replacement (since all 0 values, see SPARSITY_TURN_POINT)
		// c.reset(in.rlen, in.clen, replacement);

		if(a == null)// check for empty matrix
			return ((long) in.rlen) * in.clen;

		// overwrite with existing values (via scatter)
		for(int i = 0; i < in.rlen; i++) {
			c.fillRow(i, replacement);
			if(!a.isEmpty(i)) {
				int apos = a.pos(i);
				int cpos = c.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				double[] cvals = c.values(i);
				for(int j = apos; j < apos + alen; j++)
					if(avals[j] != 0)
						cvals[cpos + aix[j]] = avals[j];
			}
		}
		return ((long) in.rlen) * in.clen;

	}

	private static long replaceDense(MatrixBlock in, MatrixBlock ret, double pattern, double replacement) {
		DenseBlock a = in.getDenseBlock();
		DenseBlock c = ret.allocateDenseBlock().getDenseBlock();
		long nnz = 0;
		for(int bi = 0; bi < a.numBlocks(); bi++) {
			int len = a.size(bi);
			double[] avals = a.valuesAt(bi);
			double[] cvals = c.valuesAt(bi);
			for(int i = 0; i < len; i++) {
				cvals[i] = avals[i] == pattern ? replacement : avals[i];
				nnz += cvals[i] != 0 ? 1 : 0;
			}
		}
		return nnz;
	}

	private static long replaceDenseNaN(MatrixBlock in, MatrixBlock ret, double replacement) {
		DenseBlock a = in.getDenseBlock();
		DenseBlock c = ret.allocateDenseBlock().getDenseBlock();
		long nnz = 0;
		for(int bi = 0; bi < a.numBlocks(); bi++) {
			int len = a.size(bi);
			double[] avals = a.valuesAt(bi);
			double[] cvals = c.valuesAt(bi);
			for(int i = 0; i < len; i++) {
				cvals[i] = Double.isNaN(avals[i]) ? replacement : avals[i];
				nnz += cvals[i] != 0 ? 1 : 0;
			}
		}
		return nnz;

	}

}
