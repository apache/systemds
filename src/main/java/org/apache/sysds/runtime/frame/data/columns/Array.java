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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * generic, resizable native arrays
 * 
 * Base class for generic, resizable array of various value types. We use this custom class hierarchy instead of Trove
 * or other libraries in order to avoid unnecessary dependencies.
 */
public abstract class Array<T> implements Writable {
	protected static final Log LOG = LogFactory.getLog(Array.class.getName());

	protected SoftReference<HashMap<String, Long>> _rcdMapCache = null;

	protected int _size;

	protected int newSize() {
		return Math.max(_size * 2, 4);
	}

	public final SoftReference<HashMap<String, Long>> getCache() {
		return _rcdMapCache;
	}

	public final void setCache(SoftReference<HashMap<String, Long>> m) {
		_rcdMapCache = m;
	}

	public final int size() {
		return _size;
	}

	public abstract T get(int index);

	/**
	 * Get the underlying array out of the column Group, it is the responsibility of the caller to know what type it is
	 * 
	 * @return The underlying array.
	 */
	public abstract Object get();

	public abstract void set(int index, T value);

	public abstract void set(int index, double value);

	public abstract void setFromOtherType(int rl, int ru, Array<?> value);

	public abstract void set(int rl, int ru, Array<T> value);

	public abstract void set(int rl, int ru, Array<T> value, int rlSrc);

	public abstract void setNz(int rl, int ru, Array<T> value);

	public abstract void append(String value);

	public abstract void append(T value);

	@Override
	public abstract Array<T> clone();

	/**
	 * Slice out the sub range and return new array with the specified type.
	 * 
	 * If the conversion fails fallback to normal slice
	 * 
	 * @param rl row start
	 * @param ru row end
	 * @return A new array of sub range.
	 */
	public abstract Array<T> slice(int rl, int ru);

	/**
	 * Slice out the sub range and return new array with the specified type.
	 * 
	 * If the conversion fails fallback to normal slice
	 * 
	 * @param rl row start
	 * @param ru row end
	 * @param vt valuetype target
	 * @return A new array of sub range.
	 */
	public abstract Array<?> sliceTransform(int rl, int ru, ValueType vt);

	public abstract void reset(int size);

	public abstract byte[] getAsByteArray(int nRow);

	/**
	 * Get the current value type of this array.
	 * 
	 * @return The current value type.
	 */
	public abstract ValueType getValueType();

	/**
	 * Analyze the column to figure out if the value type can be refined to a better type.
	 * 
	 * @return A better or equivalent value type to represent the column.
	 */
	public abstract ValueType analyzeValueType();

	public abstract FrameArrayType getFrameArrayType();

	/**
	 * Get in memory size, not counting reference to this object.
	 * 
	 * @return the size in memory of this object.
	 */
	public abstract long getInMemorySize();

	public abstract long getExactSerializedSize();

	/**
	 * Change the allocated array to a different type. If the type is the same a deep copy is returned for safety.
	 * 
	 * @param t The type to change to
	 * @return A new column array.
	 */
	public final Array<?> changeType(ValueType t) {
		switch(t) {
			case BOOLEAN:
				return changeTypeBoolean();
			case FP32:
				return changeTypeFloat();
			case FP64:
				return changeTypeDouble();
			case UINT8:
				throw new NotImplementedException();
			case INT32:
				return changeTypeInteger();
			case INT64:
				return changeTypeLong();
			case STRING:
				return changeTypeString();
			case UNKNOWN:
			default:
				throw new DMLRuntimeException("Not a valid type to change to : " + t);
		}
	}

	protected abstract Array<?> changeTypeBoolean();

	protected abstract Array<?> changeTypeDouble();

	protected abstract Array<?> changeTypeFloat();

	protected abstract Array<?> changeTypeInteger();

	protected abstract Array<?> changeTypeLong();

	protected Array<?> changeTypeString() {
		String[] ret = new String[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i).toString();
		return new StringArray(ret);
	}

	public Pair<Integer, Integer> getMinMaxLength() {
		throw new DMLRuntimeException("Length is only relevant if case is String");
	}

	@Override
	public String toString() {
		return this.getClass().getSimpleName();
	}

}
