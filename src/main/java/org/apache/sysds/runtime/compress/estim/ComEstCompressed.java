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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;

public class ComEstCompressed extends AComEst {

	final CompressedMatrixBlock cData;

	protected ComEstCompressed(CompressedMatrixBlock data, CompressionSettings compSettings) {
		super(data, compSettings);
		cData = data;
	}

	@Override
	protected List<CompressedSizeInfoColGroup> CompressedSizeInfoColGroup(int clen, int k) {
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		final int nRow = cData.getNumRows();
		for(AColGroup g : cData.getColGroups()) {
			ret.add(g.getCompressionInfo(nRow));
		}
		return ret;
	}

	@Override
	public CompressedSizeInfoColGroup getColGroupInfo(IColIndex colIndexes, int estimate, int nrUniqueUpperBound) {
		return null;
	}

	@Override
	public CompressedSizeInfoColGroup getDeltaColGroupInfo(IColIndex colIndexes, int estimate, int nrUniqueUpperBound) {
		return null;
	}

	@Override
	protected int worstCaseUpperBound(IColIndex columns) {
		if(columns.size() == 1) {
			int id = columns.get(0);
			AColGroup g = cData.getColGroupForColumn(id);
			return g.getNumValues();
		}
		else {
			List<AColGroup> groups = CLALibCombineGroups.findGroupsInIndex(columns, cData.getColGroups());
			long nVals = 1;
			for(AColGroup g : groups)
				nVals *= g.getNumValues();

			return Math.min(_data.getNumRows(), (int) Math.min(nVals, Integer.MAX_VALUE));
		}
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
}
