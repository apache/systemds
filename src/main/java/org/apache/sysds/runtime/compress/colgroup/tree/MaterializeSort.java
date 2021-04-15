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

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.IMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MaterializeSort extends AInsertionSorter {

	public MaterializeSort(int endLength, int uniqueLabels, int knownMax) {
		super(endLength, uniqueLabels, knownMax);
	}

	@Override
	public void insert(final IntArrayList[] offsets) {
		IMapToData md = MapToFactory.create(_knownMax, _numLabels);
		md.fill(_numLabels);

		for (int i = 0; i < offsets.length; i++) {
			IntArrayList of = offsets[i];
			for (int k = 0; k < of.size(); k++)
				md.set(of.get(k), i);
		}

		int off = 0;
		for (int i = 0; i < _knownMax; i++) {
			int idx = md.getIndex(i);
			if(idx != _numLabels)
				set(off++, i, idx);
		}

	}


	@Override
	public void insert(final IntArrayList[] offsets, final int negativeIndex) {
		IMapToData md = MapToFactory.create(_knownMax, _numLabels);
		md.fill(_numLabels);
		
		for (int i = 0; i < offsets.length; i++) {
			IntArrayList of = offsets[i];
			for (int k = 0; k < of.size(); k++)
				md.set(of.get(k), i);
		}
		int off = 0;
		for (int i = 0; i < _knownMax; i++) {
			int idx = md.getIndex(i);
			if (idx < negativeIndex) 
				set(off++, i, idx);
			else if (idx > negativeIndex) 
				set(off++, i, idx - 1);
		}
	}

	@Override
	protected void insert(IntArrayList array, int label) {
		throw new DMLCompressionException("This class does not use this method");
	}

	@Override
	protected void negativeInsert(IntArrayList array) {
		throw new DMLCompressionException("This class does not use this method");
	}

	@Override
	public int[] getIndexes() {
		return _indexes;
	}

	@Override
	public IMapToData getData() {
		return _labels;
	}

}
