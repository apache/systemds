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
	private final boolean _left;
	private final boolean _right;

	public OpSided(Hop op, boolean left, boolean right) {
		super(op);
		_left = left;
		_right = right;
	}

	public boolean getLeft() {
		return _left;
	}

	public boolean getRight() {
		return _right;
	}

	@Override
	public String toString() {
		return super.toString() + " L:" + _left + " R:" + _right;
	}

	public boolean isLeftMM() {
		return !_left && _right;
	}

	public boolean isRightMM() {
		return _left && !_right;
	}

	public boolean isCompCompMM() {
		return _left && _right;
	}
}
