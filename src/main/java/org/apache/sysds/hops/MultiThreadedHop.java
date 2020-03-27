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

package org.apache.sysds.hops;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

/**
 * Optional hop interface, to be implemented by multi-threaded hops.
 */
public abstract class MultiThreadedHop extends Hop
{
	protected int _maxNumThreads = -1; //-1 for unlimited
	
	public MultiThreadedHop() {
		
	}
	
	public MultiThreadedHop(String name, DataType dt, ValueType vt) {
		super(name, dt, vt);
	}
	
	public final void setMaxNumThreads( int k ) {
		_maxNumThreads = k;
	}
	
	public final int getMaxNumThreads() {
		return _maxNumThreads;
	}
	
	//force implementing hops to make relevant
	//sub operation types as multi-threaded or not
	public abstract boolean isMultiThreadedOpType();
}
