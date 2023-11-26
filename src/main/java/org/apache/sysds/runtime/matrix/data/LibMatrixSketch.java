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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

public class LibMatrixSketch {

	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir) {
		// similar to R's unique, this operation takes a matrix and computes the unique values
		// (or rows in case of multiple column inputs)
		
		int rlen = blkIn.getNumRows();
		int clen = blkIn.getNumColumns();

		MatrixBlock blkOut = null;
		switch (dir) {
			case RowCol:
				if( clen != 1 )
					throw new NotImplementedException("Unique only support single-column vectors yet");
				// TODO optimize for dense/sparse/compressed (once multi-column support added)
				
				// obtain set of unique items (dense input vector)
				HashSet<Double> hashSet = new HashSet<>();
				for( int i=0; i<rlen; i++ ) {
					hashSet.add(blkIn.quickGetValue(i, 0));
				}
				
				// allocate output block and place values
				int rlen2 = hashSet.size();
				blkOut = new MatrixBlock(rlen2, 1, false).allocateBlock();
				Iterator<Double> iter = hashSet.iterator();
				for( int i=0; i<rlen2; i++ ) {
					blkOut.quickSetValue(i, 0, iter.next());
				}
				break;

			case Row:
				ArrayList<double[]> retainedRows = new ArrayList<>();

				for (int i=0; i<rlen; ++i) {

					// BitSet will not work because we need 2 pieces of info:
					// 1. the index and
					// 2. the value
					// A BitSet gives us only whether there is a value at a particular index, but not what that
					// specific value is.

					double[] currentRow = new double[clen];
					for (int j=0; j<clen; ++j) {
						double rawValue = blkIn.getValue(i, j);
						currentRow[j] = rawValue;
					}

					// no need to check for duplicates for the first row
					if (i == 0) {
						retainedRows.add(currentRow);
						continue;
					}

					// ensure we are not adding duplicate rows to retainedRows array
					int uniqueRowCount = 0;
					for (int m=0; m<retainedRows.size(); ++m) {

						double[] prevRow = retainedRows.get(m);

						int n = 0;
						while (n < clen) {
							if (prevRow[n] != currentRow[n]) {
								break;
							}
							n++;
						}

						// column check terminates early only if there is a column-level mismatch, ie rows are different
						if (n != clen) {
							uniqueRowCount++;
						}
					}

					// add current row to retainedRows iff it is unique from all prev retained rows
					if (uniqueRowCount == retainedRows.size()) {
						retainedRows.add(currentRow);
					}
				}

				blkOut = new MatrixBlock(retainedRows.size(), blkIn.getNumColumns(), false);
				for (int i=0; i<retainedRows.size(); ++i) {
					for (int j=0; j<blkIn.getNumColumns(); ++j) {
						blkOut.quickSetValue(i, j, retainedRows.get(i)[j]);
					}
				}

				break;

			case Col:
				throw new NotImplementedException("Unique Row/Col has not been implemented yet");

			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		return blkOut;
	}
}
