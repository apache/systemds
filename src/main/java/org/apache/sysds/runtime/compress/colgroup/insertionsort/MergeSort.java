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

public class MergeSort extends AInsertionSorter {

	private int currentFill = 0;

	public MergeSort(int endLength, int numRows, IntArrayList[] offsets) {
		super(endLength, numRows, offsets);
		insert();
	}

	public MergeSort(int endLength, int numRows, IntArrayList[] offsets, int negativeIndex) {
		super(endLength, numRows, offsets, negativeIndex);
		insertWithNegative();
	}

	private void insert() {
		for(int i = 0; i < _offsets.length; i++)
			insert(_offsets[i], i);
	}

	private void insertWithNegative() {
		for(int i = 0; i < _offsets.length; i++) {
			if(i < _negativeIndex)
				insert(_offsets[i], i);
			else if(i > _negativeIndex)
				insert(_offsets[i], i - 1);
		}
		negativeInsert(_offsets[_negativeIndex]);
	}

	protected void insert(IntArrayList array, int label) {
		if(currentFill == 0) {
			currentFill = array.size();
			for(int i = 0; i < currentFill; i++)
				set(i, array.get(i), label);
		}
		else
			merge(array, label);
	}

	private void merge(IntArrayList a, int label) {
		int pA = a.size(); // Pointer A
		int pP = currentFill; // Pointer Previous
		currentFill = pA + pP;
		int pN = currentFill - 1; // Pointer new
		pA--; // last element
		pP--; // last element
		int vA, vP;

		while(pP >= 0 && pA >= 0) {
			vA = a.get(pA);
			vP = _indexes[pP];
			if(vP > vA) {
				set(pN--, vP, _labels.getIndex(pP--));
			}
			else {
				set(pN--, vA, label);
				pA--;
			}
		}
		while(pA >= 0)
			set(pN--, a.get(pA--), label);
	}

	protected void negativeInsert(IntArrayList a) {
		final int label = _numLabels - 1;
		int pA = a.size() - 1; // Pointer A
		int pP = currentFill - 1; // Pointer Previous
		// From here on currentFill is no longer needed.
		int pN = _indexes.length - 1; // Pointer new
		int vA = a.get(pA);
		// Pointer to last index
		int vM = _numRows - 1;

		// While both old indexes have to be added and a have to be ignored.
		while(pP >= 0 && pA >= 0 && pN >= 0) {
			final int vP = _indexes[pP];
			vA = a.get(pA);
			if(vP == vM)
				set(pN--, vM, _labels.getIndex(pP--));
			else if(vA == vM)
				pA--;
			else
				set(pN--, vM, label);
			vM--;
		}

		// If there is no more indexes to ignore
		if(pA < 0) {
			// add all remaining indexes from other arrays
			while(pP >= 0 && pN >= 0) {
				final int vP = _indexes[pP];
				if(vP == vM)
					set(pN--, vM, _labels.getIndex(pP--));
				else
					set(pN--, vM, label);
				vM--;
			}
		}
		else {
			// skip all indexes in a.
			while(pN >= 0 && pA >= 0) {
				vA = a.get(pA);
				if(vA < vM)
					set(pN--, vM, label);
				else
					pA--;
				vM--;
			}
		}

		// Fill the rest with the default value.
		while(pN >= 0 && vM >= 0)
			set(pN--, vM--, label);
	}
}
