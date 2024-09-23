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

import org.apache.sysds.common.Types;

import java.util.HashSet;

public class LibMatrixSketch {

	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir) {
		//similar to R's unique, this operation takes a matrix and computes the
		//unique values (or rows in case of multiple column inputs)
		
		int rlen = blkIn.getNumRows();
		int clen = blkIn.getNumColumns();

		MatrixBlock blkOut = null;
		// TODO optimize for dense/sparse/compressed (once multi-column support added)
		
		switch (dir) {
			case RowCol: {
				// obtain set of unique items (dense input vector)
				HashSet<Double> hashSet = new HashSet<>();
				for( int i=0; i<rlen; i++ ) {
					for( int j=0; j<clen; j++ )
						hashSet.add(blkIn.get(i, j));
				}
				
				// allocate output block and place values
				int rlen2 = hashSet.size();
				blkOut = new MatrixBlock(rlen2, 1, false).allocateBlock();
				int pos = 0;
				for( Double val : hashSet )
					blkOut.set(pos++, 0, val);
				break;
			}
			case Row: {
				//2-pass algorithm to avoid unnecessarily large mem requirements
				HashSet<Double> hashSet = new HashSet<>();
				int clen2 = 0;
				for( int i=0; i<rlen; i++ ) {
					hashSet.clear();
					for( int j=0; j<clen; j++ )
						hashSet.add(blkIn.get(i, j));
					clen2 = Math.max(clen2, hashSet.size());
				}
				
				//actual 
				blkOut = new MatrixBlock(rlen, clen2, false).allocateBlock();
				for( int i=0; i<rlen; i++ ) {
					hashSet.clear();
					for( int j=0; j<clen; j++ )
						hashSet.add(blkIn.get(i, j));
					int pos = 0;
					for( Double val : hashSet )
						blkOut.set(i, pos++, val);
				}
				break;
			}
			case Col: {
				//2-pass algorithm to avoid unnecessarily large mem requirements
				HashSet<Double> hashSet = new HashSet<>();
				int rlen2 = 0;
				for( int j=0; j<clen; j++ ) {
					hashSet.clear();
					for( int i=0; i<rlen; i++ )
						hashSet.add(blkIn.get(i, j));
					rlen2 = Math.max(rlen2, hashSet.size());
				}
				
				//actual 
				blkOut = new MatrixBlock(rlen2, clen, false).allocateBlock();
				for( int j=0; j<clen; j++ ) {
					hashSet.clear();
					for( int i=0; i<rlen; i++ )
						hashSet.add(blkIn.get(i, j));
					int pos = 0;
					for( Double val : hashSet )
						blkOut.set(pos++, j, val);
				}
				break;
			}
			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		return blkOut;
	}
}
