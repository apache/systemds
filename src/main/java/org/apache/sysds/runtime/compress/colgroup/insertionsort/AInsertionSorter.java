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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * This abstract class is for sorting the IntArrayList entries efficiently for SDC Column Groups construction.
 * 
 * The idea is to construct an object, where the array is inserted along with a label, and the values are sorted at
 * insertion time.
 * 
 */
public abstract class AInsertionSorter {
	protected static final Log LOG = LogFactory.getLog(AInsertionSorter.class.getName());

	protected final IntArrayList[] _offsets;
	protected final int _negativeIndex;

	protected final int[] _indexes;
	protected final AMapToData _labels;

	protected final int _numLabels;
	protected final int _numRows;

	public AInsertionSorter(int endLength, int numRows, IntArrayList[] offsets) {
		_indexes = new int[endLength];
		_numLabels = offsets.length;
		_labels = MapToFactory.create(endLength, _numLabels);
		_numRows = numRows;
		_offsets = offsets;
		_negativeIndex = -1;
	}

	public AInsertionSorter(int endLength, int numRows, IntArrayList[] offsets, int negativeIndex) {
		_indexes = new int[endLength];
		_numLabels = offsets.length;
		_labels = MapToFactory.create(endLength, _numLabels);
		_numRows = numRows;
		_offsets = offsets;
		_negativeIndex = negativeIndex;
	}

	public int[] getIndexes() {
		return _indexes;
	}

	public AMapToData getData() {
		return _labels;
	}

	protected void set(int index, int value, int label) {
		_indexes[index] = value;
		_labels.set(index, label);
	}
}
