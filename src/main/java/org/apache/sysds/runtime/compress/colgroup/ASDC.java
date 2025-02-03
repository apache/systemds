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

import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.SDCScheme;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns. Then the zeros
 * would be materialized in the group without any overhead.
 */
public abstract class ASDC extends AMorphingMMColGroup implements AOffsetsGroup, IContainDefaultTuple {
	private static final long serialVersionUID = 769993538831949086L;

	/** Sparse row indexes for the data */
	protected final AOffset _indexes;
	/** The number of rows in this column group */
	protected final int _numRows;

	protected ASDC(IColIndex colIndices, int numRows, IDictionary dict, AOffset offsets, int[] cachedCounts) {
		super(colIndices, dict, cachedCounts);
		_indexes = offsets;
		_numRows = numRows;
	}

	public int getNumRows() {
		return _numRows;
	}

	@Override
	public AOffset getOffsets() {
		return _indexes;
	}

	public abstract int getNumberOffsets();

	@Override
	public final CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		EstimationFactors ef = new EstimationFactors(getNumValues(), _numRows, getNumberOffsets(), _dict.getSparsity());
		return new CompressedSizeInfoColGroup(_colIndexes, ef, estimateInMemorySize(), getCompType(), getEncoding());
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return SDCScheme.create(this);
	}

	@Override
	public AColGroup morph(CompressionType ct, int nRow) {
		if(ct == getCompType())
			return this;
		else if(ct == CompressionType.SDCFOR)
			return this; // it does not make sense to change to FOR.
		else
			return super.morph(ct, nRow);
	}

	@Override
	protected boolean allowShallowIdentityRightMult() {
		return false;
	}
}
