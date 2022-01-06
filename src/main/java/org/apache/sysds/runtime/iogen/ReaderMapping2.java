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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;

public class ReaderMapping2 {

	private int[][] mapRow;
	private int[][] mapCol;
	private boolean symmetric;
	private boolean skewSymmetric;
	private boolean isUpperTriangular;
	private int skewCoefficient;
	private ArrayList<RawIndex> sampleRawIndexes;

	private boolean mapped;
	private static int nrows;
	private static int ncols;
	private int nlines;
	private int firstRowIndex;
	private int firstColIndex;

	private MatrixBlock sampleMatrix;
	private FrameBlock sampleFrame;
	private Types.ValueType[] schema;
	private final boolean isMatrix;

	public ReaderMapping2(String raw, MatrixBlock matrix) throws Exception {
		this.ReadRaw(raw);
		this.isMatrix = true;
		this.sampleMatrix = matrix;
		this.nrows = this.sampleMatrix.getNumRows();
		this.ncols = this.sampleMatrix.getNumColumns();
		this.runMapping();
	}

	public ReaderMapping2(String raw, FrameBlock frame) throws Exception {
		this.ReadRaw(raw);
		this.isMatrix = false;
		this.sampleFrame = frame;
		this.nrows = this.sampleFrame.getNumRows();
		this.ncols = this.sampleFrame.getNumColumns();
		this.schema = this.sampleFrame.getSchema();
		this.runMapping();
	}

	private void ReadRaw(String raw) throws Exception {
		this.sampleRawIndexes = new ArrayList<>();
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		int nlines = 0;

		while((value = br.readLine()) != null) {
			this.sampleRawIndexes.add(new RawIndex(value));
			nlines++;
		}
		this.nlines = nlines;
		this.firstColIndex = 0;
		this.firstRowIndex = 0;
	}

	private boolean isSchemaNumeric() {
		if(isMatrix)
			return true;

		boolean result = true;
		for(Types.ValueType vt : schema)
			result &= vt.isNumeric();
		return result;
	}

	private void runMapping() throws Exception {
		mapped = findMapping();
	}

	protected boolean findMapping() {
		mapRow = new int[nrows][ncols];
		mapCol = new int[nrows][ncols];

		// Set "-1" as default value for all defined matrix
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				mapRow[r][c] = mapCol[r][c] = -1;

		int itRow = 0;
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if((this.isMatrix && this.sampleMatrix.getValue(r, c) != 0) || (!this.isMatrix && this.sampleFrame.get(
					r, c) != null)) {
					HashSet<Integer> checkedLines = new HashSet<>();
					while(checkedLines.size() < nlines) {
						RawIndex ri = sampleRawIndexes.get(itRow);
						Pair<Integer, Integer> pair = this.isMatrix ? ri.findValue(
							sampleMatrix.getValue(r, c)) : ri.findValue(sampleFrame.get(r, c), schema[c]);
						if(pair != null) {
							mapRow[r][c] = itRow;
							mapCol[r][c] = pair.getKey();
							break;
						}
						else {
							checkedLines.add(itRow);
							itRow++;
							if(itRow == nlines)
								itRow = 0;
						}
					}
				}
			}
		}
		boolean flagMap = true;
		for(int r = 0; r < nrows && flagMap; r++)
			for(int c = 0; c < ncols && flagMap; c++)
				if(mapRow[r][c] == -1 && ((!this.isMatrix && this.sampleFrame.get(r,
					c) != null) || (this.isMatrix && this.sampleMatrix.getValue(r, c) != 0))) {
					flagMap = false;
				}
		return flagMap;
	}

	public CustomProperties2 getFormatProperties() {
		CustomProperties2 properties = new CustomProperties2();

		// Find Row Index Properties
		// 1. is row index identified?
		boolean rowIndexIdentify = isRowIndexIdentify();
		if(!rowIndexIdentify) {
			Pair<String, Boolean> pair = isRowIndexPrefix();
			if(pair==null){

			}
			else {
				properties.setRowIndex(CustomProperties2.IndexProperties.PREFIX);
				properties.setRowIndexPrefixDelim(pair.getKey());
				properties.setRowIndexPrefixDelimFixLength(pair.getValue());
			}
		}
		else
			properties.setRowIndex(CustomProperties2.IndexProperties.IDENTIFY);

		return properties;
	}

	private boolean isRowIndexIdentify() {
		int l = 0;
		ArrayList<Pair<Integer, Integer>> mismatched = new ArrayList<>();
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] != -1 && l != mapRow[r][c]) {
					mismatched.add(new Pair<>(r, c));
				}
			}
			l++;
		}
		// All rows of sample raw not used
		if(l != nlines) {
			return false;
		}
		else if(mismatched.size() > 0) {
			return false;
		}
		return true;
	}

	private Pair<String, Boolean> isRowIndexPrefix() {

		ArrayList<Pair<Integer, Integer>> mismatched = new ArrayList<>();
		ArrayList<Pair<Integer, Integer>> prefixes = new ArrayList<>();
		ArrayList<Pair<Integer, Integer>> nonePrefix = new ArrayList<>();
		DelimiterTrie delimiterTrie = new DelimiterTrie();

		int delimiterMinSize = 0;

		for(int r = 0; r < nrows; r++) {
			RawIndex ri = sampleRawIndexes.get(r);
			ri.cloneReservedPositions();
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] != -1) {
					Pair<Integer, Integer> pair = ri.findValue(r);
					if(pair == null)
						mismatched.add(new Pair<>(r, c));
					else {
						if(pair.getKey() < mapCol[r][c]) {
							String delim = ri.getSubString(pair.getKey() + pair.getValue(), mapCol[r][c]);
							int delimLength = delim.length();
							if(delimiterMinSize != 0 && delimLength < delimiterMinSize)
								delimiterMinSize = delimLength;
							else
								delimiterMinSize = delimLength;

							delimiterTrie.insert(delim);
							prefixes.add(pair);
						}
						else
							nonePrefix.add(pair);
					}
				}
			}
			//ri.restoreReservedPositions();
		}
		// TODO: attend to mistakes and none-prefix row index maps

		return delimiterTrie.getShortestDelim(delimiterMinSize);
	}

	class DelimiterTrie {
		private final StringBuilder totalDelim;
		private int totalDelimLength;
		private boolean valid;

		public DelimiterTrie() {
			totalDelim = new StringBuilder();
			totalDelimLength = 0;
			valid = true;
		}

		public boolean insert(String delim) {
			if(delim.length() > totalDelimLength) {
				if(delim.startsWith(totalDelim.toString())) {
					totalDelim.append(delim.substring(totalDelimLength));
					totalDelimLength += delim.length() - totalDelimLength;
				}
				else
					valid = false;
			}
			else if(!totalDelim.toString().startsWith(delim))
				valid = false;
			return valid;
		}

		public Pair<String, Boolean> getShortestDelim(int minsize) {
			if(!valid)
				return null;

			if(minsize == totalDelimLength)
				return new Pair<String, Boolean>(totalDelim.toString(), true);
			else {
				HashSet<String> delimSet = new HashSet<>();
				for(int i = 1; i <= minsize; i++) {
					delimSet.clear();
					for(int j = 0; j < totalDelimLength; j += i) {
						delimSet.add(totalDelim.substring(j, Math.min(j + i, totalDelimLength)));
					}
					if(delimSet.size() == 1)
						break;
				}
				if(delimSet.size() == 1) {
					String delim = delimSet.iterator().next();
					return new Pair<String, Boolean>(delim, delim.length() == totalDelimLength);
				}
				else
					return null;
			}
		}

		public void print() {
			System.out.println(totalDelim);
		}
	}


	public boolean isMapped() {
		return mapped;
	}
}
