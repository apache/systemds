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
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;

public class ComEstCompressedSample extends ComEstSample {

	private static boolean loggedWarning = false;


	public ComEstCompressedSample(CompressedMatrixBlock sample, CompressionSettings cs, CompressedMatrixBlock full,
		int k) {
		super(sample, cs, full, k);
		// cData = sample;
	}

	@Override
	protected List<CompressedSizeInfoColGroup> CompressedSizeInfoColGroup(int clen, int k) {
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		final int nRow = _data.getNumRows();
		final List<AColGroup> fg = ((CompressedMatrixBlock) _data).getColGroups();
		final List<AColGroup> sg = ((CompressedMatrixBlock) _sample).getColGroups();

		for(int i = 0; i < fg.size(); i++) {
			CompressedSizeInfoColGroup r = fg.get(i).getCompressionInfo(nRow);
			r.setMap(sg.get(i).getCompressionInfo(_sampleSize).getMap());
			ret.add(r);
		}

		return ret;
	}

	@Override
	public CompressedSizeInfoColGroup getColGroupInfo(IColIndex colIndexes, int estimate, int nrUniqueUpperBound) {
		if(!loggedWarning)
			LOG.warn("Compressed input cannot fallback to resampling " + colIndexes);
		loggedWarning = true;
		return null;
	}

	@Override
	public CompressedSizeInfoColGroup getDeltaColGroupInfo(IColIndex colIndexes, int estimate, int nrUniqueUpperBound) {
		if(!loggedWarning)
			LOG.warn("Compressed input cannot fallback to resampling " + colIndexes);
		return null;
	}

	@Override
	protected int worstCaseUpperBound(IColIndex columns) {
		CompressedMatrixBlock cData = ((CompressedMatrixBlock) _data);
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

}
