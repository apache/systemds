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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Abstract class for column group types that do not perform matrix Multiplication, and decompression for performance
 * reasons but instead transforms into another type of column group type to perform that operation.
 */
public abstract class AMorphingMMColGroup extends AColGroupValue {
	private static final long serialVersionUID = -4265713396790607199L;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected AMorphingMMColGroup(int numRows) {
		super(numRows);
	}

	/**
	 * A Abstract class for column groups that contain ADictionary for values.
	 * 
	 * @param colIndices   The Column indexes
	 * @param numRows      The number of rows contained in this group
	 * @param dict         The dictionary to contain the distinct tuples
	 * @param cachedCounts The cached counts of the distinct tuples (can be null since it should be possible to
	 *                     reconstruct the counts on demand)
	 */
	protected AMorphingMMColGroup(int[] colIndices, int numRows, ADictionary dict, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
	}

	@Override
	protected final void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	protected final void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	protected final void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	protected final void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public final void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public final void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public final void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	protected final void tsmm(double[] result, int numColumns, int nRows) {
		throw new DMLCompressionException("This method should never be called");
	}

	public abstract AColGroup extractCommon(double[] constV);
}
