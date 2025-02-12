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

package org.apache.sysds.runtime.compress.estim;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.encoding.EmptyEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Exact compressed size estimator (examines entire dataset).
 */
public class ComEstExact extends AComEst {

	public ComEstExact(MatrixBlock data, CompressionSettings compSettings) {
		super(data, compSettings);
	}

	@Override
	public CompressedSizeInfoColGroup getColGroupInfo(IColIndex colIndexes, int estimate, int nrUniqueUpperBound) {
		final IEncode map = EncodingFactory.createFromMatrixBlock(_data, _cs.transposed, colIndexes, _cs.scaleFactors);
		if(map instanceof EmptyEncoding)
			return new CompressedSizeInfoColGroup(colIndexes, getNumRows(), CompressionType.EMPTY);
		return getFacts(map, colIndexes);
	}

	@Override
	public CompressedSizeInfoColGroup getDeltaColGroupInfo(IColIndex colIndexes, int estimate, int nrUniqueUpperBound) {
		final IEncode map = EncodingFactory.createFromMatrixBlockDelta(_data, _cs.transposed, colIndexes);
		return getFacts(map, colIndexes);
	}

	@Override
	protected CompressedSizeInfoColGroup combine(IColIndex combinedColumns, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int maxDistinct) {
		final IEncode map = g1.getMap().combine(g2.getMap());
		return getFacts(map, combinedColumns);
	}

	protected CompressedSizeInfoColGroup getFacts(IEncode map, IColIndex colIndexes) {
		final int _numRows = getNumRows();
		final EstimationFactors em = map.extractFacts(_numRows, _data.getSparsity(), _data.getSparsity(), _cs);
		return new CompressedSizeInfoColGroup(colIndexes, em, _cs.validCompressions, map);
	}

	@Override
	protected int worstCaseUpperBound(IColIndex columns) {
		return getNumRows();
	}

}
