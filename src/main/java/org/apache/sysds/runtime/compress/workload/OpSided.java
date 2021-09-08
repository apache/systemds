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

package org.apache.sysds.runtime.compress.workload;

import org.apache.sysds.hops.Hop;

public class OpSided extends Op {

	// Compressed Sides:
	private final boolean _cLeft;
	private final boolean _cRight;
	// Transposed Sides:
	private final boolean _tLeft;
	private final boolean _tRight;

	public OpSided(Hop op, boolean cLeft, boolean cRight, boolean tLeft, boolean tRight) {
		super(op);
		_cLeft = cLeft;
		_cRight = cRight;
		_tLeft = tLeft;
		_tRight = tRight;
	}

	public boolean getLeft() {
		return _cLeft;
	}

	public boolean getRight() {
		return _cRight;
	}

	public boolean getTLeft() {
		return _tLeft;
	}

	public boolean getTRight() {
		return _tRight;
	}

	@Override
	public String toString() {
		return super.toString() + " L:" + _cLeft + " R:" + _cRight + " tL:" + _tLeft + " tR:" + _tRight + " ";
	}

	public boolean isLeftMM() {
		return (!_cLeft && _cRight && !_tRight) || (_cLeft && !_cRight && _tLeft);
	}

	public boolean isRightMM() {
		return (_cLeft && !_cRight && !_tLeft) || (!_cLeft && _cRight && _tRight);
	}

	public boolean isCompCompMM() {
		return _cLeft && _cRight;
	}

	@Override
	public boolean isCompressedOutput() {
		// if the output is transposed after a right matrix multiplication the compression is decompressed
		return _cLeft && !_cRight && !_tLeft;
	}

}
