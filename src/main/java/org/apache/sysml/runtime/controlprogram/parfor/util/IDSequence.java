/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

/**
 * ID sequence for generating unique long identifiers with start 0 and increment 1.
 * 
 */
public class IDSequence 
{

	
	private long _current = -1;
	private boolean wrapAround = false;
	
	public IDSequence()
	{
		reset();
	}
	
	public IDSequence(boolean wrapAround)
	{
		reset();
		this.wrapAround = wrapAround;
	}
	
	/**
	 * Creates the next ID, if overflow a RuntimeException is thrown.
	 * 
	 * @return ID
	 */
	public synchronized long getNextID()
	{
		_current++;
		
		if( _current == Long.MAX_VALUE ) {
			if (wrapAround)
				reset();
			else
				throw new RuntimeException("WARNING: IDSequence will produced numeric overflow.");
		}
		
		return _current;
	}
	
	public synchronized void reset()
	{
		_current = 0;
	}
	
	/*
	private AtomicLong _seq = new AtomicLong(0);
	
	public long getNextID()
	{
		return _seq.getAndIncrement();
	}
	
	public void reset()
	{
		_seq = new AtomicLong( 0 );
	}
	*/
}
