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

package org.apache.sysds.runtime.controlprogram.parfor.util;

import java.util.concurrent.atomic.AtomicLong;

/**
 * ID sequence for generating unique long identifiers with start 0 and increment 1.
 * 
 */
public class IDSequence 
{
	private final AtomicLong _current;
	private final boolean _cyclic;
	private final long _cycleLen;
	
	public IDSequence() {
		this(false, -1);
	}
	
	public IDSequence(boolean cyclic) {
		this(cyclic, Long.MAX_VALUE);
	}
	
	public IDSequence(boolean cyclic, long cycleLen) {
		_current = new AtomicLong(-1);
		_cyclic = cyclic;
		_cycleLen = cycleLen;
	}
	
	
	/**
	 * Creates the next ID, if overflow a RuntimeException is thrown.
	 * 
	 * @return ID
	 */
	public long getNextID() {
		long val = _current.incrementAndGet();
		if( val == _cycleLen ) {
			if( !_cyclic )
				throw new RuntimeException("WARNING: IDSequence will produced numeric overflow.");
			reset();
		}
		return val;
	}
	
	public long getCurrentID() {
		return _current.get();
	}
	
	public void reset() {
		reset(-1);
	}
	
	public void reset(long value) {
		_current.set(value);
	}
}
