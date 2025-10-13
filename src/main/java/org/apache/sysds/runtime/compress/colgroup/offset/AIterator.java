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
package org.apache.sysds.runtime.compress.colgroup.offset;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Iterator interface, that returns a iterator of the indexes while also maintaining a data index.
 */
public abstract class AIterator {
	public static final Log LOG = LogFactory.getLog(AIterator.class.getName());

	protected int offset;

	/**
	 * Main Constructor
	 * 
	 * @param offset The current offset into in the uncompressed representation.
	 */
	protected AIterator(int offset) {
		this.offset = offset;
	}

	/**
	 * Increment the pointers such that the both index and dataIndex is incremented to the next entry.
	 * 
	 * @return The new offset.
	 */
	public abstract int next();

	/**
	 * Get the current index value, note this correspond to a row index in the original matrix.
	 * 
	 * @return The current value pointed at.
	 */
	public int value() {
		return offset;
	}

	public void setOff(int off) {
		offset = off;
	}

	/**
	 * Find out if the current offset is not exceeding the index given.
	 * 
	 * @param ub The offset to not exceed
	 * @return boolean if it is exceeded.
	 */
	public boolean isNotOver(int ub) {
		return offset < ub;
	}

	/**
	 * Get the current data index associated with the index returned from value.
	 * 
	 * This index points to a position in the AMapToData object, that can be used to lookup the dictionary entry in
	 * ADictionary.
	 * 
	 * @return The Data Index.
	 */
	public abstract int getDataIndex();

	/**
	 * Get the current offsets index, that points to the underlying offsets list.
	 * 
	 * This is available for debugging purposes, not to be used for the calling classes.
	 * 
	 * @return The Offsets Index.
	 */
	public abstract int getOffsetsIndex();

	/**
	 * Skip values until index is achieved.
	 * 
	 * @param idx The index to skip to.
	 * @return the index that follows or are equal to the skip to index.
	 */
	public abstract int skipTo(int idx);

	/**
	 * Copy the iterator with the current values.
	 */
	public abstract AIterator clone();

	/**
	 * Unsafe version of equals, note that it should only compare iterators stemming from the same Offset Object.
	 * 
	 * @param o The Iterator to compare
	 * @return The result
	 */
	public boolean equals(AIterator o) {
		return o.getOffsetsIndex() == getOffsetsIndex();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" v:" + value() + " d:" + getDataIndex() + " o:" + getOffsetsIndex());
		return sb.toString();
	}
}
