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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
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
	}

	private ColGroupDeltaDDC(int[] colIndexes, ADictionary dict, AMapToData data, int[] cachedCounts) {
		super();
		LOG.info("Carefully use of DeltaDDC since implementation is not finished.");
		_colIndexes = colIndexes;
		_dict = dict;
		_data = data;
	}

	public static AColGroup create(int[] colIndices, ADictionary dict, AMapToData data, int[] cachedCounts) {
		if(dict == null)
			throw new NotImplementedException("Not implemented constant delta group");
		else
			return new ColGroupDeltaDDC(colIndices, dict, data, cachedCounts);
	}

	public CompressionType getCompType() {
		return CompressionType.DeltaDDC;
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			final int rowIndex = _data.getIndex(i) * nCol;
			final int prevOff = (off == 0) ? off : off - nCol;
			for(int j = 0; j < nCol; j++) {
				// Here we use the values in the previous row to compute current values along with the delta
				double newValue = c[prevOff + j] + values[rowIndex + j];
				c[off + _colIndexes[j]] += newValue;
			}
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupDeltaDDC(_colIndexes, _dict.applyScalarOp(op), _data, getCachedCounts());
	}
}
