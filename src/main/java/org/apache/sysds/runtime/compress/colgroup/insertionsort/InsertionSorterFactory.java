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

import org.apache.sysds.runtime.compress.utils.IntArrayList;

public final class InsertionSorterFactory {

	public enum SORT_TYPE {
		MERGE, MATERIALIZE;
	}

	public static AInsertionSorter create(int knownMax, IntArrayList[] offsets) {
		return create(getEndLength(offsets), knownMax, offsets);
	}

	public static AInsertionSorter create(int endLength, int knownMax, IntArrayList[] offsets) {
		return create(endLength, knownMax, offsets, -1, SORT_TYPE.MATERIALIZE);
	}

	public static AInsertionSorter create(int knownMax, IntArrayList[] offsets, int negativeIndex) {
		return create(getEndLength(offsets) - offsets[negativeIndex].size(), knownMax, offsets, negativeIndex);
	}

	public static AInsertionSorter create(int endLength, int knownMax, IntArrayList[] offsets, int negativeIndex) {
		return create(endLength, knownMax, offsets, negativeIndex, SORT_TYPE.MATERIALIZE);
	}

	public static AInsertionSorter create(int knownMax, IntArrayList[] offsets, int negativeIndex, SORT_TYPE st) {
		return create(getEndLength(offsets) - offsets[negativeIndex].size(), knownMax, offsets, negativeIndex, st);
	}

	public static AInsertionSorter create(int endLength, int knownMax, IntArrayList[] offsets, int negativeIndex,
		SORT_TYPE st) {
		switch(st) {
			case MERGE:
				return new MergeSort(endLength, knownMax, offsets, negativeIndex);
			default:
				return new MaterializeSort(endLength, knownMax, offsets, negativeIndex);
		}
	}

	private static int getEndLength(IntArrayList[] offsets) {
		int endLength = 0;
		for(IntArrayList l : offsets) {
			endLength += l.size();
		}
		return endLength;
	}
}
