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

import java.util.Iterator;

/**
 * General-purpose iterator to decode a compressed OLE bitmap.
 * 
 */
public final class BitmapDecoderRLE implements Iterator<Integer>
{
	// pointer to the compressed bitmap
	private int _bmOff;
	private int _bmLen;
	private char[] _bmPtr;

	// The offset of the <b>next</b> element we will read from the bitmap, or
	// bmPtr.length if we are done.
	private int _nextBmOffset;

	// The offset in the matrix column of the beginning of the current run
	private int _runStartOffset;

	// The length of the current run
	private int _curRunLen;

	// The number of bits that we have returned from the current run.
	private int _runBitsReturned;

	/**
	 * Point this object at the beginning of a particular bitmap. After a call
	 * to this method, the next call to {@link #nextOffset()} will return the
	 * offset of the first bit in the specified bitmap.
	 * 
	 * @param bmPtr
	 *            pointer to a compressed bitmap
	 */
	public BitmapDecoderRLE(char[] bmPtr, int off, int len) {
		_bmOff = off;
		_bmLen = len;
		_bmPtr = bmPtr;
		_nextBmOffset = 0;
		_runStartOffset = 0;
		_curRunLen = 0;
		_runBitsReturned = 0;

		if (0 == _bmLen) {
			return; //no runs
		}

		// Advance to the beginning of the first non-empty run.
		advanceToNextRun();
	}

	@Override
	public Integer next() {
		if( !hasNext() )
			throw new RuntimeException("No next offset existing.");
		
		// Grab the lookahead value
		int ret = _runStartOffset + _runBitsReturned;

		_runBitsReturned++;

		// Check for end of run
		if (_runBitsReturned == _curRunLen) {
			advanceToNextRun();
		}

		return ret;
	}

	@Override
	public boolean hasNext() {
		return _runBitsReturned < _curRunLen;
	}
	
	@Override
	public void remove() {
		throw new RuntimeException("Not implemented for BitmapDecoderRLE.");
	}
	
	/** Move forward to the next non-empty run. */
	private void advanceToNextRun() {
		// While loop needed because some runs are of length 0
		while (_runBitsReturned == _curRunLen && _nextBmOffset < _bmLen) {

			_runBitsReturned = 0;

			// Read the distance to the next run
			char delta = _bmPtr[_bmOff + _nextBmOffset];

			// Run length is stored in the next element of the array
			_runStartOffset += delta + _curRunLen;
			_curRunLen = _bmPtr[_bmOff + _nextBmOffset + 1];
			_nextBmOffset += 2;
		}
	}
}
