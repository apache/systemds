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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is first delta encoded then encoded with dense dictionary
 * encoding (DeltaDDC).
 */
public class ColGroupDeltaDDC extends ColGroupDDC {
	private static final long serialVersionUID = -1045556313148564147L;

	/** Constructor for serialization */
	protected ColGroupDeltaDDC() {
		super();
	}

	private ColGroupDeltaDDC(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndexes, dict, data, cachedCounts);
		if(CompressedMatrixBlock.debug) {
			if(!(dict instanceof DeltaDictionary))
				throw new DMLCompressionException("DeltaDDC must use DeltaDictionary");
		}
	}

	public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		if(data.getUnique() == 1)
			return ColGroupConst.create(colIndexes, dict);
		else if(dict == null)
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupDeltaDDC(colIndexes, dict, data, cachedCounts);
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.DeltaDDC;
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.size();
		final double[] prevRow = new double[nCol];
		
		if(rl > 0) {
			final double[] prevRowData = db.values(rl - 1 + offR);
			final int prevOff = db.pos(rl - 1 + offR) + offC;
			for(int j = 0; j < nCol; j++) {
				prevRow[j] = prevRowData[prevOff + _colIndexes.get(j)];
			}
		}

		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			final int dictIdx = _data.getIndex(i);
			final int rowIndex = dictIdx * nCol;
			
			if(i == 0 && rl == 0) {
				for(int j = 0; j < nCol; j++) {
					final double value = values[rowIndex + j];
					final int colIdx = _colIndexes.get(j);
					c[off + colIdx] = value;
					prevRow[j] = value;
				}
			}
			else {
				for(int j = 0; j < nCol; j++) {
					final double delta = values[rowIndex + j];
					final double newValue = prevRow[j] + delta;
					final int colIdx = _colIndexes.get(j);
					c[off + colIdx] = newValue;
					prevRow[j] = newValue;
				}
			}
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new NotImplementedException("Sparse block decompression for DeltaDDC not yet implemented");
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		if(_dict instanceof DeltaDictionary) {
			DeltaDictionary deltaDict = (DeltaDictionary) _dict;
			IDictionary newDict = deltaDict.applyScalarOp(op);
			return new ColGroupDeltaDDC(_colIndexes, newDict, _data, getCachedCounts());
		}
		else {
			throw new DMLRuntimeException("DeltaDDC must use DeltaDictionary");
		}
	}
}
