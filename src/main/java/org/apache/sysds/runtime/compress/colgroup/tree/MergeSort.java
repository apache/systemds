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

import org.apache.sysds.runtime.compress.colgroup.mapping.IMapToData;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MergeSort extends AInsertionSorter {

	private int currentFill = 0;

	public MergeSort(int endLength, int uniqueLabels, int knownMax) {
		super(endLength, uniqueLabels, knownMax);
	}

	@Override
	protected void insert(IntArrayList array, int label) {

		if (currentFill == 0) {
			currentFill = array.size();
			for (int i = 0; i < currentFill; i++)
				set(i, array.get(i), label);
		} else
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

		while (pP >= 0 && pA >= 0) {
			vA = a.get(pA);
			vP = _indexes[pP];
			if (vP > vA) {
				set(pN--, vP, _labels.getIndex(pP--));
			} else {
				set(pN--, vA, label);
				pA--;
			}
		}
		while (pA >= 0)
			set(pN--, a.get(pA--), label);

	}


	@Override
	protected void negativeInsert(IntArrayList a) {
		if (currentFill == _indexes.length)
			return;
		final int label = _numLabels - 1;
		int pA = a.size() - 1; // Pointer A
		int pP = currentFill - 1; // Pointer Previous
		// From here on currentFill is no longer needed.
		int pN = _indexes.length - 1; // Pointer new
		int vA = a.get(pA);
		int vP;
		int vM = _knownMax;
		while (pP > 0 && pA >= 0 && pN >= 0) {
			vP = _indexes[pP];
			vA = a.get(pA);
			if (vP == vM)
				set(pN--, vM, _labels.getIndex(pP--));
			else if (vA == vM)
				pA--;
			else
				set(pN--, vM, label);
			vM--;
		}

		while (pN >= 0 && pA >= 0) {
			vA = a.get(pA);
			if (vA < vM)
				set(pN--, vM, label);
			else if (vA == vM)
				pA--;

			vM--;

		}

		while (pN >= 0 && vM > 0)
			set(pN--, vM--, label);

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
