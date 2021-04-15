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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.IMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * This abstract class is for sorting the IntArrayList entries efficiently for
 * SDC Column Groups construction.
 * 
 * The idea is to construct an insertion tree, where the array is inserted along
 * with a label, and the values are sorted at insertion time.
 * 
 * Any implementations is guaranteed that calls to getIndexes is first done once
 * all _indexes are assigned.
 * 
 */
public abstract class AInsertionSorter {
	protected static final Log LOG = LogFactory.getLog(AInsertionSorter.class.getName());

	protected final int[] _indexes;
	protected final IMapToData _labels;

	protected final int _numLabels;
	protected final int _knownMax;

	public AInsertionSorter(int endLength, int uniqueLabels, int knownMax) {
		_indexes = new int[endLength];
		_labels = MapToFactory.create(endLength, uniqueLabels);
		_numLabels = uniqueLabels;
		_knownMax = knownMax;
	}

	public void insert(IntArrayList[] offsets) {
		for (int i = 0; i < offsets.length; i++) {
			insert(offsets[i], i);
		}
	}

	public void insert(IntArrayList[] offsets, int negativeIndex) {
		for (int i = 0; i < offsets.length; i++) {
			if (i < negativeIndex)
				insert(offsets[i], i);
			else if (i > negativeIndex)
				insert(offsets[i], i - 1);
		}
		if (offsets[negativeIndex].size() > 0)
			negativeInsert(offsets[negativeIndex]);
	}

	protected abstract void insert(IntArrayList array, int label);

	/**
	 * This method is to insert the remaining entries that are missing. But the
	 * trick is that the array provided is the entries that are not to be assigned.
	 * But instead any positions that are missing in the current _indexes and not
	 * present in the provided array.
	 * 
	 * This method should only be called after all other arrays are inserted.
	 * 
	 * @param array The provided array that should not be inserted, but instead the
	 *              "negative imprint" of it should.
	 */
	protected abstract void negativeInsert(IntArrayList array);

	public abstract int[] getIndexes();

	public abstract IMapToData getData();

	protected void set(int index, int value, int label) {
		_indexes[index] = value;
		_labels.set(index, label);
	}
}
