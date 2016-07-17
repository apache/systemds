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

package org.apache.sysml.runtime.compress;

import java.util.Arrays;
import java.util.Iterator;

/**
 * General-purpose iterator to decode a compressed OLE bitmap.
 *  
 */
public final class BitmapDecoderOLE implements Iterator<Integer> 
{
	// pointer to the compressed bitmap
	private int _bmOff;
	private int _bmLen;
	private char[] _bmPtr;

	// The index of the current block. Block 0 covers bits 1 through 2^16
	private int _blockIx;

	// The offset where the current block starts within the bitmap.
	private int _blockStartOffset;

	// The number of offsets in the current block.
	private int _curBlockSize;

	// The offset <b>in the current block</b> the <b>next</b> element we will
	// read from the bitmap, or bmPtr.length if we are done.
	private int _nextBmOffset;

	/**
	 * Point this object at the beginning of a particular bitmap. After a call
	 * to this method, the next call to {@link #nextOffset()} will return the
	 * offset of the first bit in the specified bitmap.
	 * 
	 * @param bmPtr
	 *            pointer to a compressed bitmap
	 */
	public BitmapDecoderOLE(char[] bmPtr, int off, int len) {
		_bmOff = off;
		_bmLen = len;
		_bmPtr = bmPtr;
		_blockIx = 0;
		_blockStartOffset = 0;
		_curBlockSize = _bmPtr[_bmOff+_blockStartOffset];
		if (_curBlockSize < 0) {
			throw new RuntimeException(String.format(
					"Negative block size %d at position %d of %s",
					_curBlockSize, _blockStartOffset, Arrays.toString(bmPtr)));
		}
		_nextBmOffset = 0;

		// Advance past any zero-length blocks at the beginning of the array
		while (_blockStartOffset < _bmLen
				&& _nextBmOffset >= _curBlockSize) {
			advanceToNextBlock();
		}
	}

	@Override
	public Integer next() {
		if( !hasNext() )
			throw new RuntimeException("No next offset existing.");
		
		// Grab the lookahead value Note the +1 in the array indexing; 
		// the first number in a block is the block size
		int offsetFromBlockBegin = _bmPtr[_bmOff+_blockStartOffset + 1 + _nextBmOffset];
		int ret = (_blockIx * BitmapEncoder.BITMAP_BLOCK_SZ)
				+ offsetFromBlockBegin;
		_nextBmOffset++;

		// Advance to next non-empty block if we reached the end of the block.
		while (_blockStartOffset < _bmLen && _nextBmOffset >= _curBlockSize) {
			advanceToNextBlock();
		}

		return ret;
	}

	@Override
	public boolean hasNext() {
		return _blockStartOffset < _bmLen;
	}

	@Override
	public void remove() {
		throw new RuntimeException("Not implemented for BitmapDecoderOLE.");
	}

	/**
	 * Move forward to the next block. Does not skip empty blocks.
	 */
	private void advanceToNextBlock() {
		_blockStartOffset += (1 + _curBlockSize);
		_blockIx++;
		if (_blockStartOffset >= _bmLen) {
			// Read past last block
			return;
		}

		_curBlockSize = _bmPtr[_bmOff+_blockStartOffset];
		if (_curBlockSize < 0) {
			throw new RuntimeException(String.format(
					"Negative block size %d at position %d of %s",
					_curBlockSize, _blockStartOffset, Arrays.toString(_bmPtr)));
		}
		_nextBmOffset = 0;
	}
}
