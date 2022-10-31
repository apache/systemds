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

package org.apache.sysds.runtime.frame.data.columns;

import java.lang.ref.SoftReference;
import java.util.HashMap;

import org.apache.hadoop.io.Writable;
import org.apache.sysds.common.Types.ValueType;

/**
 * generic, resizable native arrays
 * 
 * Base class for generic, resizable array of various value types. We use this custom class hierarchy instead of Trove
 * or other libraries in order to avoid unnecessary dependencies.
 */
public abstract class Array<T> implements Writable {
	protected SoftReference<HashMap<String, Long>> _rcdMapCache = null;

	protected int _size = 0;

	protected int newSize() {
		return Math.max(_size * 2, 4);
	}

	public abstract T get(int index);

	/**
	 * Get the underlying array out of the column Group, it is the responsibility of the caller to know what type it is
	 * 
	 * @return The underlying array.
	 */
	public abstract Object get();

	public final SoftReference<HashMap<String, Long>> getCache(){
		return _rcdMapCache;
	}

	public final void setCache(SoftReference<HashMap<String, Long>> m){
		_rcdMapCache = m;
	}

	public final int size(){
		return _size;
	}

	public abstract void set(int index, T value);

	public abstract void set(int rl, int ru, Array<T> value);

	public abstract void set(int rl, int ru, Array<T> value, int rlSrc);

	public abstract void setNz(int rl, int ru, Array<T> value);

	public abstract void append(String value);

	public abstract void append(T value);

	@Override
	public abstract Array<T> clone();

	public abstract Array<T> slice(int rl, int ru);

	public abstract void reset(int size);

	public abstract byte[] getAsByteArray(int nRow);

	public abstract ValueType getValueType();
	
	@Override
	public String toString() {
		return this.getClass().getSimpleName().toString() + ":" + _size;
	}

}
