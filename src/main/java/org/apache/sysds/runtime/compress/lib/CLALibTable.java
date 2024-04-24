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

package org.apache.sysds.runtime.compress.lib;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CLALibTable {

	protected static final Log LOG = LogFactory.getLog(CLALibTable.class.getName());

	private CLALibTable() {
		// empty constructor
	}

	public static MatrixBlock tableSeqOperations(int seqHeight, MatrixBlock A, int nColOut) {

		final int[] map = new int[seqHeight];
		boolean containsNull = false; // figure out if there are nulls.
		int maxCol = 0;

		for(int i = 0; i < seqHeight; i++) {
			final double v2 = A.get(i, 0);
			if(Double.isNaN(v2)) {
				map[i] = -1; // assign temporarily to -1
				containsNull = true;
			}
			else {
				// safe casts to long for consistent behavior with indexing
				int col = UtilFunctions.toInt(v2);
				if(col <= 0)
					throw new DMLRuntimeException(
						"Erroneous input while computing the contingency table (value <= zero): " + v2);

				map[i] = col - 1;
				// maintain max seen col
				maxCol = Math.max(col, maxCol);
			}
		}

		if(nColOut == -1)
			nColOut = maxCol;

		if(containsNull) { // correct for null.
			for(int i = 0; i < seqHeight; i++) {
				if(map[i] == -1)
					map[i] = nColOut;
			}
		}

		if(nColOut == 0) // edge case of empty zero dimension block.
			return new MatrixBlock(seqHeight, 0, 0.0);

		// create a single DDC Column group.
		final IColIndex i = ColIndexFactory.create(0, nColOut);
		final ADictionary d = new IdentityDictionary(nColOut, containsNull);
		final AMapToData m = MapToFactory.create(seqHeight, map, nColOut + (containsNull ? 1 : 0));
		final AColGroup g = ColGroupDDC.create(i, d, m, null);

		final CompressedMatrixBlock cmb = new CompressedMatrixBlock(seqHeight, nColOut);
		cmb.allocateColGroup(g);
		cmb.recomputeNonZeros();

		return cmb;
	}
}
