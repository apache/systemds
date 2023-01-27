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

package org.apache.sysds.runtime.compress.colgroup.indexes;

import java.io.DataOutput;
import java.io.IOException;

/**
 * Class to contain column indexes for the Compression column groups.
 */
public interface IColIndex {

	public static enum ColIndexType {
		SINGLE, TWO, ARRAY;
	}

	/**
	 * Get the size of the index aka, how many columns is contained
	 * 
	 * @return The size
	 */
	public int size();

	/**
	 * Get the index at a specific location
	 * 
	 * @param i The index to get
	 * @return the column index at the index.
	 */
	public int get(int i);

	/**
	 * Return a new column index where the values are shifted by the specified amount.
	 * 
	 * It is returning a new instance of the index.
	 * 
	 * @param i The amount to shift
	 * @return the new instance of an index.
	 */
	public IColIndex shift(int i);

	public void write(DataOutput out) throws IOException;

	public long getExactSizeOnDisk();

	public long estimateInMemorySize();

	/**
	 * A Iterator of the indexes see the iterator interface for details.
	 * 
	 * @return A iterator for the indexes contained.
	 */
	public IIterate iterator();

}
