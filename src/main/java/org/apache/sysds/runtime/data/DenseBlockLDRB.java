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


package org.apache.sysds.runtime.data;


import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.stream.IntStream;

/**
 * Dense Large Row Blocks have multiple 1D arrays (blocks), which contain complete rows.
 * Except the last block all blocks have the same size (size refers to the number of rows contained and space allocated).
 */
public abstract class DenseBlockLDRB extends DenseBlock
{
	private static final long serialVersionUID = -7519435549328146356L;

	protected int _blen;

	protected DenseBlockLDRB(int[] dims) {
		super(dims);
	}

	/**
	 * Create the internal array to store the blocks. Does not create
	 * storage space for a block yet, call allocate block for that.
	 *
	 * @param numBlocks the number of blocks to create
	 */
	protected abstract void allocateBlocks(int numBlocks);

	@Override
	public int blockSize() {
	    return _blen;
	}

	@Override
	public int blockSize(int bix) {
		return Math.min(_blen, _rlen - bix * _blen);
	}

	@Override
	public void reset(int rlen, int[] odims, double v) {
		long dataLength = (long) rlen * odims[0];
		int newBlockSize = Math.min(rlen, Integer.MAX_VALUE / odims[0]);
		int numBlocks = UtilFunctions.toInt(Math.ceil((double) rlen / newBlockSize));
		if (_blen == newBlockSize && dataLength <= capacity()) {
			IntStream.range(0, numBlocks)
					.forEach((bi) -> {
						int toIndex = (int)Math.min(newBlockSize, dataLength - bi * newBlockSize) * _odims[0];
						fillBlock(bi, 0, toIndex, v);
					});
		} else {
			int lastBlockSize = (newBlockSize == rlen ? newBlockSize : rlen % newBlockSize) * odims[0];
			allocateBlocks(numBlocks);
			IntStream.range(0, numBlocks)
					.forEach((i) -> {
						int length = (i == numBlocks - 1 ? lastBlockSize : newBlockSize *  _odims[0]);
						allocateBlock(i, length);
						if (v != 0)
							fillBlock(i, 0, length, v);
					});
		}
		_blen = newBlockSize;
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public int pos(int[] ix) {
		int pos = pos(ix[0]);
		pos += ix[ix.length - 1];
		for(int i = 1; i < ix.length - 1; i++)
			pos += ix[i] * _odims[i];
		return pos;
	}

	@Override
	public boolean isContiguous(int rl, int ru) {
		return index(rl) == index(ru);
	}

	@Override
	public int size(int bix) {
		return blockSize(bix) * _odims[0];
	}

	@Override
	public int index(int r) {
		return r / blockSize();
	}

	@Override
	public int pos(int r) {
		return (r % blockSize()) * _odims[0];
	}

	@Override
	public int pos(int r, int c) {
		return (r % blockSize()) * _odims[0] + c;
	}

	@Override
	public long countNonZeros() {
		long nnz = 0;
		for (int i = 0; i < numBlocks() - 1; i++) {
			nnz += computeNnz(i, 0, blockSize() * _odims[0]);
		}
		return nnz + computeNnz(numBlocks() - 1, 0, blockSize(numBlocks() - 1) * _odims[0]);
	}

	@Override
	public int countNonZeros(int r) {
		return (int) computeNnz(index(r), pos(r), _odims[0]);
	}

	@Override
	public long countNonZeros(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		boolean allColumns = cl == 0 && cu == _odims[0];
		int rb = pos(rl);
		int re = blockSize() * _odims[0];
		// loop over rows of blocks, and call computeNnz for the specified columns
		for (int bi = index(rl); bi <= index(ru - 1); bi++) {
			// loop complete block if not last one
			if (bi == index(ru - 1)) {
				re = pos(ru - 1) + _odims[0];
			}
			if (allColumns) {
				nnz += computeNnz(bi, rb, re - rb);
			} else {
				for (int ri = rb; ri < re; ri += _odims[0]) {
					nnz += computeNnz(bi, ri + cl, cu - cl);
				}
			}
			rb = 0;
		}
		return nnz;
	}

	@Override
	public DenseBlock set(double v) {
		for (int i = 0; i < numBlocks() - 1; i++) {
			fillBlock(i, 0, blockSize() * _odims[0], v);
		}
		fillBlock(numBlocks() - 1, 0, blockSize(numBlocks() - 1) * _odims[0], v);
		return this;
	}

	@Override
	public DenseBlock set(int rl, int ru, int cl, int cu, double v) {
		boolean allColumns = cl == 0 && cu == _odims[0];
		int rb = pos(rl);
		int re = blockSize() * _odims[0];
		for (int bi = index(rl); bi <= index(ru - 1); bi++) {
			if (bi == index(ru - 1)) {
				re = pos(ru - 1) + _odims[0];
			}
			if (allColumns) {
				fillBlock(bi, rb, re, v);
			}
			else {
				for (int ri = rb; ri < re; ri += _odims[0]) {
					fillBlock(bi, ri + cl, ri + cu, v);
				}
			}
			rb = 0;
		}
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		int bix = index(r);
		int offset = pos(r);
		IntStream.range(0, _odims[0])
				.forEach((i) -> setInternal(bix, offset + i, v[i]));
		return this;
	}

	@Override
	public DenseBlock set(DenseBlock db) {
		// ToDo: Optimize if dense block types match
		// ToDo: Performance
		for (int ri = 0; ri < _rlen; ri += blockSize()) {
			int bix = ri / blockSize();
			double[] other = db.valuesAt(bix);
			IntStream.range(0, blockSize(bix) * _odims[0])
					.forEach((i) -> setInternal(bix, i, other[i]));
		}
		return this;
	}
}
