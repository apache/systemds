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
 * Iterator interface that only iterate through offsets without considering data index.
 */
public abstract class AOffsetIterator {
	public static final Log LOG = LogFactory.getLog(AOffsetIterator.class.getName());

	protected int offset;

	/**
	 * Main Constructor
	 * 
	 * @param offset The current offset into in the uncompressed representation.
	 */
	protected AOffsetIterator(int offset) {
		this.offset = offset;
	}

	/**
	 * Increment the pointer and return the new offset gained
	 * 
	 * @return The new offset.
	 */
	public abstract int next();

	/**
	 * Get the current index value, note this correspond to a row index in the original matrix.
	 * 
	 * @return The current value pointed at.
	 */
	public final int value() {
		return offset;
	}
}
