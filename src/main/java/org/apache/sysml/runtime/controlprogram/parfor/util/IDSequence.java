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

package org.apache.sysml.runtime.controlprogram.parfor.util;

import java.util.concurrent.atomic.AtomicLong;

/**
 * ID sequence for generating unique long identifiers with start 0 and increment 1.
 * 
 */
public class IDSequence 
{
	private final AtomicLong _current;
	private final boolean _wrapAround;
	
	public IDSequence() {
		this(false);
	}
	
	public IDSequence(boolean wrapAround) {
		_current = new AtomicLong(-1);
		_wrapAround = wrapAround;
	}
	
	/**
	 * Creates the next ID, if overflow a RuntimeException is thrown.
	 * 
	 * @return ID
	 */
	public long getNextID()
	{
		long val = _current.incrementAndGet();
		
		if( val == Long.MAX_VALUE ) {
			if( !_wrapAround )
				throw new RuntimeException("WARNING: IDSequence will produced numeric overflow.");
			reset();
		}
		
		return val;
	}
	
	public long getCurrentID() {
		return _current.get();
	}
	
	public void reset() {
		_current.set(0);
	}
}
