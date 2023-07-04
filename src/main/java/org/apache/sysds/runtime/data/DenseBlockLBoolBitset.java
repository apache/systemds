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

import java.util.BitSet;
import java.util.stream.IntStream;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Warnings;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DenseBlockLBoolBitset extends DenseBlockLDRB
{
	private static final long serialVersionUID = 2604223782138590322L;

	private BitSet[] _blocks;

	public DenseBlockLBoolBitset(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	@Override
	protected void allocateBlocks(int numBlocks) {
		_blocks = new BitSet[numBlocks];
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_blocks[bix] = new BitSet(length);
	}

	@Override
	protected void setInternal(int bix, int ix, double v) {
		_blocks[bix].set(ix, v != 0);
	}

	@Override
	public boolean isNumeric() {
		return true;
	}
	
	@Override
	public boolean isNumeric(ValueType vt) {
		return ValueType.BOOLEAN == vt;
	}


	@Override
	public boolean isContiguous() {
		return _blocks.length == 1;
	}

	@Override
	public void reset(int rlen, int[] odims, double v) {
		// Special implementation to make computeNnz fast if complete block is read
		boolean bv = v != 0;
		long dataLength = (long) rlen * odims[0];
		int newBlockSize = Math.min(rlen, MAX_ALLOC / odims[0]);
		int numBlocks = UtilFunctions.toInt(Math.ceil((double) rlen / newBlockSize));
		if (_blen == newBlockSize && dataLength <= capacity()) {
			for (int i = 0; i < numBlocks; i++) {
				int toIndex = (int)Math.min(newBlockSize, dataLength - i * newBlockSize) * _odims[0];
				_blocks[i].set(0, toIndex, bv);
				// Clear old data so we can use cardinality for computeNnz
				_blocks[i].set(toIndex, _blocks[i].size(), false);
			}
		} else {
			int lastBlockSize = (newBlockSize == rlen ? newBlockSize : rlen % newBlockSize)  * odims[0];
			allocateBlocks(numBlocks);
			IntStream.range(0, numBlocks)
					.forEach((i) -> {
						int length = i == numBlocks - 1 ? lastBlockSize : newBlockSize;
						allocateBlock(i, length);
						_blocks[i].set(0, length, bv);
					});
		}
		_blen = newBlockSize;
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public int numBlocks() {
		return _blocks.length;
	}

	@Override
	public long capacity() {
		return (_blocks!=null) ? (long)(_blocks.length - 1) * _blocks[0].size() + _blocks[_blocks.length - 1].size() : -1;
	}

	@Override
	protected long computeNnz(int bix, int start, int length) {
		if (start == 0 && length == blockSize(bix) * _odims[0]) {
			return _blocks[bix].cardinality();
		} else {
			BitSet mask = new BitSet(_blocks[bix].size());
			mask.set(start, length + start);
			mask.and(_blocks[bix]);
			return mask.cardinality();
		}
	}

	@Override
	public double[] values(int r) {
		return valuesAt(index(r));
	}
	
	@Override
	public double[] valuesAt(int bix) {
		int length = blockSize(bix) * _odims[0];
		Warnings.warnFullFP64Conversion(length);
		return DataConverter.toDouble(_blocks[bix], length);
	}

	@Override
	public void incr(int r, int c) {
		_blocks[index(r)].set(pos(r, c));
	}

	@Override
	public void incr(int r, int c, double delta) {
		if (delta != 0) {
			_blocks[index(r)].set(pos(r, c));
		}
	}

	@Override
	protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		_blocks[bix].set(fromIndex, toIndex, v != 0);
	}

	@Override
	public DenseBlock set(String s) {
		boolean b = Boolean.parseBoolean(s);
		for (int i = 0; i < numBlocks() - 1; i++) {
			_blocks[i].set(0, blockSize() *_odims[0], b);
		}
		_blocks[numBlocks() - 1].set(0, blockSize(numBlocks() - 1) * _odims[0], b);
		return this;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		_blocks[index(r)].set(pos(r, c), v != 0);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, double v) {
		_blocks[index(ix[0])].set(pos(ix), v != 0);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, long v) {
		_blocks[index(ix[0])].set(pos(ix), v != 0);
		return this;
	}
	@Override
	public DenseBlock set(int[] ix, String v) {
		_blocks[index(ix[0])].set(pos(ix), Boolean.parseBoolean(v));
		return this;
	}

	@Override
	public double get(int r, int c) {
		return _blocks[index(r)].get(pos(r, c)) ? 1 : 0;
	}

	@Override
	public double get(int[] ix) {
		return _blocks[index(ix[0])].get(pos(ix)) ? 1 : 0;
	}

	@Override
	public String getString(int[] ix) {
		return String.valueOf(_blocks[index(ix[0])].get(pos(ix)));
	}

	@Override
	public long getLong(int[] ix) {
		return _blocks[index(ix[0])].get(pos(ix)) ? 1 : 0;
	}
}
