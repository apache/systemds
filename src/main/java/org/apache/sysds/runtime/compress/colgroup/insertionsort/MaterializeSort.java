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

package org.apache.sysds.runtime.compress.colgroup.insertionsort;

import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MaterializeSort extends AInsertionSorter {

	/** a dense mapToData, that have a value for each row in the input. */
	AMapToData md;

	public MaterializeSort(int endLength, int knownMax, IntArrayList[] offsets, int negativeIndex) {
		super(endLength, knownMax, offsets, negativeIndex);

		md = MapToFactory.create(_knownMax, _numLabels);
		md.fill(_numLabels);
		if(_negativeIndex == -1)
			insert();
		else
			insertWithNegative();
	}

	private void insert() {

		for(int i = 0; i < _offsets.length; i++) {
			IntArrayList of = _offsets[i];
			for(int k = 0; k < of.size(); k++)
				md.set(of.get(k), i);
		}

		int off = 0;
		for(int i = 0; i < _knownMax; i++) {
			int idx = md.getIndex(i);
			if(idx != _numLabels)
				set(off++, i, idx);
		}

	}

	private void insertWithNegative() {
		for(int i = 0; i < _offsets.length; i++) {
			IntArrayList of = _offsets[i];
			for(int k = 0; k < of.size(); k++)
				md.set(of.get(k), i);
		}
		int off = 0;
		for(int i = 0; i < _knownMax; i++) {
			int idx = md.getIndex(i);
			if(idx < _negativeIndex)
				set(off++, i, idx);
			else if(idx > _negativeIndex)
				set(off++, i, idx - 1);
		}
	}

}
