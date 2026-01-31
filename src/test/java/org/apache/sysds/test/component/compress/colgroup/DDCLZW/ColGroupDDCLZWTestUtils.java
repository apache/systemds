package org.apache.sysds.test.component.compress.colgroup.DDCLZW;/*
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

import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;

import java.util.Arrays;

/// This class contains static methods to generate mappings and DDCs for tests/benchmarks for ColGroupDDCLZW
public class ColGroupDDCLZWTestUtils {
	/**
	 * Creates a sample DDC group for unit tests from a given mapping
	 */
	public static ColGroupDDC createDDC(int[] mapping, int nUnique, int nCols) {
		IColIndex colIndexes = ColIndexFactory.create(nCols);

		double[] dictValues = new double[nUnique * nCols];
		for(int i = 0; i < nUnique; i++) {
			for(int c = 0; c < nCols; c++) {
				dictValues[i * nCols + c] = (i + 1) * 10.0 + c;
			}
		}
		Dictionary dict = Dictionary.create(dictValues);

		AMapToData data = MapToFactory.create(mapping.length, nUnique);
		for(int i = 0; i < mapping.length; i++) {
			data.set(i, mapping[i]);
		}

		return (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
	}

	// Pattern generators (array)
	public static int[] genPatternRepeating(int size, int... pattern) {
		int[] result = new int[size];
		for(int i = 0; i < size; i++) {
			result[i] = pattern[i % pattern.length];
		}
		return result;
	}

	/// Args (10, 5) generates a pattern like: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
	public static int[] genPatternDistributed(int size, int nUnique) {
		int[] result = new int[size];
		int runLength = size / nUnique;
		int pos = 0;
		for(int i = 0; i < nUnique && pos < size; i++) {
			int endPos = Math.min(pos + runLength, size);
			Arrays.fill(result, pos, endPos, i);
			pos = endPos;
		}
		return result;
	}

	public static int[] genPatternRandom(int size, int nUnique, long seed) {
		int[] result = new int[size];
		java.util.Random rand = new java.util.Random(seed);
		for(int i = 0; i < size; i++) {
			result[i] = rand.nextInt(nUnique);
		}
		return result;
	}

	/// Args (10, 3) generates a pattern like: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
	public static int[] genPatternLZWOptimal(int size, int nUnique) {
		int[] result = new int[size];
		for(int i = 0; i < size; i++) {
			result[i] = i % nUnique;
		}
		return result;
	}
}
