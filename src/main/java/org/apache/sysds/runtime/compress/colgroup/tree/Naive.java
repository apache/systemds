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

package org.apache.sysds.runtime.compress.colgroup.tree;

import java.util.Arrays;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class Naive extends AInsertionSorter {

	public Naive(int endLength, int uniqueLabels, int knownMax) {
		super(endLength, uniqueLabels, knownMax);
	}

	@Override
	public void insert(final IntArrayList[] offsets) {
		

		int[] offsetsRowIndex = new int[offsets.length];

		int lastIndex = -1;

		for(int i = 0; i < _indexes.length; i++) {
			lastIndex++;
			int start = Integer.MAX_VALUE;
			int groupId = Integer.MAX_VALUE;
			for(int j = 0; j < offsets.length; j++) {
				final int off = offsetsRowIndex[j];
				if(off < offsets[j].size()) {
					final int v = offsets[j].get(off);
					if(v == lastIndex) {
						start = lastIndex;
						groupId = j;
						break;
					}
					else if(v < start) {
						start = v;
						groupId = j;
					}
				}
			}
			offsetsRowIndex[groupId]++;
			_labels.set(i, groupId);
			_indexes[i] = start;
			lastIndex = start;
		}

		if(_indexes[_indexes.length-1] == 0 )
			throw new DMLCompressionException("Invalid Index Structure" + Arrays.toString(_indexes));


	}


	@Override
	public void insert(final IntArrayList[] offsets, final int negativeIndex) {
		
		final int[] offsetsRowIndex = new int[offsets.length];

		boolean isJ;
		for(int i = 0, k = 0; i < _knownMax; i++) {
			int beforeK = k;
			isJ = false;
			for(int j = 0; j < offsets.length; j++) {
				final int off = offsetsRowIndex[j];
				if(off < offsets[j].size()) {
					final int v = offsets[j].get(off);
					if(v == i) {
						if(j != negativeIndex) {
							_labels.set(k, j - (j < negativeIndex ? 0 : 1));
							_indexes[k] = i;
							k++;
						}
						else
							isJ = true;

						offsetsRowIndex[j]++;
						break;
					}
				}
			}
			if(!isJ && beforeK == k) { // materialize zeros.
				_labels.set(k, offsets.length - 1);
				_indexes[k] = i;
				k++;
			}
		}

		if(_indexes[_indexes.length -1] == 0){
			LOG.error(Arrays.toString(offsets));
			throw new DMLCompressionException("Invalid! " + Arrays.toString(_indexes));
		}


	}

	@Override
	protected void insert(IntArrayList array, int label) {
		throw new DMLCompressionException("This naive method does not use this method");
	}

	@Override
	protected void negativeInsert(IntArrayList array) {
		throw new DMLCompressionException("This naive method does not use this method");
	}

	@Override
	public int[] getIndexes() {
		return _indexes;
	}

	@Override
	public AMapToData getData() {
		return _labels;
	}

}
