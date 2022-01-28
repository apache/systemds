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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MaterializeSort extends AInsertionSorter {

	/** The block size to materialize at a time */
	public static int CACHE_BLOCK = 16000;

	/** a dense mapToData, that have a value for each row in the input. */
	private final AMapToData md;
	private final int[] skip;

	private final int placeholder;
	private int off = 0;

	protected MaterializeSort(int endLength, int numRows, IntArrayList[] offsets) {
		super(endLength, numRows, offsets);
		placeholder = _numLabels + 1;
		// + 1 to ensure that the _numLabels is exceeded.
		md = MapToFactory.create(Math.min(_numRows, CACHE_BLOCK), Math.max(placeholder, 3));
		skip = new int[offsets.length];
		for(int block = 0; block < _numRows; block += CACHE_BLOCK)
			insert(block, Math.min(block + CACHE_BLOCK, _numRows));
		
	}

	protected MaterializeSort(int endLength, int numRows, IntArrayList[] offsets, int negativeIndex) {
		super(endLength, numRows, offsets, negativeIndex);

		placeholder = _numLabels;
		md = MapToFactory.create(Math.min(_numRows, CACHE_BLOCK), Math.max(placeholder, 3));
		skip = new int[offsets.length];

		for(int block = 0; block < _numRows; block += CACHE_BLOCK) 
			insertWithNegative(block, Math.min(block + CACHE_BLOCK, _numRows));
		
	}

	private void insert(int rl, int ru) {
		try {
			md.fill(placeholder);
			materializeInsert(rl, ru);
			filterInsert(rl, ru);
		}
		catch(Exception e) {
			int sum = 0;
			for(IntArrayList o : _offsets)
				sum += o.size();
			throw new DMLCompressionException("Failed normal materialize sorting with list of " + _offsets.length + " with sum (aka output size): " + sum + " requested Size: " + _indexes.length + " range: " + rl + " " + ru , e);
		}
	}

	private void materializeInsert(int rl, int ru) {
		for(int i = 0; i < _offsets.length; i++) {
			final IntArrayList of = _offsets[i];
			final int size = of.size();
			int k = skip[i];
			while(k < size && of.get(k) < ru)
				md.set(of.get(k++) - rl, i);
			skip[i] = k;
		}
	}

	private void filterInsert(int rl, int ru) {
		final int len = ru - rl;
		for(int i = 0; i < len; i++) {
			final int idx = md.getIndex(i);
			if(idx != placeholder)
				set(off++, i + rl, idx);
		}
	}

	private void insertWithNegative(int rl, int ru) {
		md.fill(placeholder);

		for(int i = 0; i < _offsets.length; i++) {
			IntArrayList of = _offsets[i];
			int k = skip[i];
			while(k < of.size() && of.get(k) < ru)
				md.set(of.get(k++) - rl, i);
			skip[i] = k;
		}

		for(int i = rl; i < ru; i++) {
			final int idx = md.getIndex(i - rl);
			if(idx < _negativeIndex)
				set(off++, i, idx);
			else if(idx > _negativeIndex)
				set(off++, i, idx - 1);
		}
	}

}
