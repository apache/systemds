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

package org.apache.sysds.runtime.compress.colgroup;

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns. Then the zeros
 * would be materialized in the group without any overhead.
 */
public abstract class ASDC extends AMorphingMMColGroup {
	private static final long serialVersionUID = 769993538831949086L;

	/** Sparse row indexes for the data */
	protected AOffset _indexes;
	
	final protected int _numRows;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ASDC(int numRows) {
		super();
		_numRows = numRows;
	}

	protected ASDC(int[] colIndices, int numRows, ADictionary dict,  AOffset offsets,
		int[] cachedCounts) {
		super(colIndices, dict, cachedCounts);

		_indexes = offsets;
		_numRows = numRows;
	}

	public int getNumRows(){
		return _numRows;
	}

	public abstract double[] getDefaultTuple();

	public AOffset getOffsets(){
		return _indexes;
	}
}
