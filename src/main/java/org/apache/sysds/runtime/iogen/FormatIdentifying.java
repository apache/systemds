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

package org.apache.sysds.runtime.iogen;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.util.ArrayList;
import java.util.HashSet;

public class FormatIdentifying {

	private int[][] mapRow;
	private int[][] mapCol;
	private int[][] mapLen;
	private ArrayList<RawIndex> sampleRawIndexes;

	private static int nrows;
	private static int ncols;
	private int nlines;
	private final boolean isMatrix;
	private int colIndexBeginFrom;
	private int rowIndexBeginFrom;

	private ReaderMapping mappingValues;
	private CustomProperties properties;

	public FormatIdentifying(String raw, MatrixBlock matrix) throws Exception {
		this.mappingValues = new ReaderMapping(raw, matrix);
		this.isMatrix = true;
		this.runIdentification();
	}

	public FormatIdentifying(String raw, FrameBlock frame) throws Exception {
		this.mappingValues = new ReaderMapping(raw, frame);
		this.isMatrix = false;
		this.runIdentification();
	}

	private void runIdentification() {

		mapRow = mappingValues.getMapRow();
		mapCol = mappingValues.getMapCol();
		mapLen = mappingValues.getMapLen();
		sampleRawIndexes = mappingValues.getSampleRawIndexes();

		nrows = mappingValues.getNrows();
		ncols = mappingValues.getNcols();
		nlines = mappingValues.getNlines();

		Pair<ArrayList<String>[], HashSet<String>[]> patternPair = buildKeyPattern();
		properties = new CustomProperties(patternPair.getKey(), patternPair.getValue());
		properties.setRowIndex(CustomProperties.IndexProperties.IDENTIFY);
	}


	public CustomProperties getFormatProperties() {
		return properties;
	}

	private Pair<ArrayList<String>[], HashSet<String>[]> buildKeyPattern() {

		ArrayList<String>[] colKeys = new ArrayList[ncols];
		HashSet<String>[] colKeyEndWithValueStrings = new HashSet[ncols];
		for(int c = 0; c < ncols; c++) {
			Pair<ArrayList<String>, HashSet<String>> pair = buildKeyPatternForAColumn(c);
			if(pair != null) {
				colKeys[c] = pair.getKey();
				colKeyEndWithValueStrings[c] = pair.getValue();
			}
			else {
				return null;
			}
		}
		return new Pair<>(colKeys, colKeyEndWithValueStrings);
	}

	private Pair<ArrayList<String>, HashSet<String>> buildKeyPatternForAColumn(int colIndex) {
		ArrayList<Pair<String, Integer>> prefixStringAndLineNumber = getAllPrefixStringsOfAColumn(colIndex);
		MappingTrie trie = new MappingTrie();
		for(Pair<String, Integer> p : prefixStringAndLineNumber) {
			trie.reverseInsert(p.getKey(), p.getValue());
		}
		ArrayList<ArrayList<String>> keys = trie.getAllSequentialKeys();
		HashSet<String> endWithValueString = null;
		boolean flagReconstruct;
		int selectedIndex = -1;

		do {
			int index = 0;
			for(ArrayList<String> key : keys) {
				endWithValueString = verifyColKeyInALine(colIndex, key);
				if(endWithValueString != null) {
					selectedIndex = index;
					break;
				}
				index++;
			}
			if(endWithValueString == null) {
				flagReconstruct = trie.reConstruct();

				if(flagReconstruct)
					keys = trie.getAllSequentialKeys();
				else
					break;
			}
			else
				break;

		}
		while(true);

		if(selectedIndex != -1)
			return new Pair<>(keys.get(selectedIndex), endWithValueString);
		else
			return null;
	}

	// Get all prefix strings of a column
	private ArrayList<Pair<String, Integer>> getAllPrefixStringsOfAColumn(int colIndex) {
		ArrayList<Pair<String, Integer>> prefixStringAndLineNumber = new ArrayList<>();
		int rowIndex;
		for(int r = 0; r < nrows; r++) {
			rowIndex = mapRow[r][colIndex];
			if(rowIndex != -1) {
				prefixStringAndLineNumber.add(new Pair<>(
					sampleRawIndexes.get(rowIndex).getSubString(0, mapCol[r][colIndex]), rowIndex));
			}
		}
		return prefixStringAndLineNumber;
	}

	// Validate a key in a row of sample raw data
	private HashSet<String> verifyColKeyInALine(int colIndex, ArrayList<String> key) {

		boolean flag = true;
		HashSet<String> endWithValueString = new HashSet<>();
		for(int r = 0; r < nrows; r++) {
			int rowIndex = mapRow[r][colIndex];
			if(rowIndex != -1) {
				RawIndex ri = sampleRawIndexes.get(rowIndex);
				int currPos = 0;
				for(String k : key) {
					int index = ri.getRaw().indexOf(k, currPos);
					if(index != -1)
						currPos = index + k.length();
					else {
						flag = false;
						break;
					}
				}
				int endDelimPos = mapCol[r][colIndex] + mapLen[r][colIndex];
				endWithValueString.add(ri.getSubString(endDelimPos, Math.min(endDelimPos + 1, ri.getRawLength())));
				if(!flag || currPos != mapCol[r][colIndex]) {
					return null;
				}
			}
		}
		if(endWithValueString.size() == 0)
			return null;
		return endWithValueString;
	}
}
