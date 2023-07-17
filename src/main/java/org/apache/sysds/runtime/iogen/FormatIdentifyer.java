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

import org.apache.spark.sql.sources.In;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class FormatIdentifyer {
	private int[][] mapRow;
	private int[][] mapCol;
	private int[][] mapLen;
	private int actualValueCount;
	private MappingProperties mappingProperties;
	private RawIndex[] sampleRawIndexes;
	private int nrows;
	private int ncols;
	private int nlines;
	private int suffixStringLength = 50;
	private ReaderMapping mappingValues;
	private CustomProperties properties;
	private BitSet staticColIndexes;

	public FormatIdentifyer(String raw, MatrixBlock matrix) throws Exception {
		this.mappingValues = new ReaderMapping(raw, matrix);
		this.runIdentification();
	}

	public FormatIdentifyer(String raw, FrameBlock frame) throws Exception {
		this.mappingValues = new ReaderMapping(raw, frame);
		this.runIdentification();
	}

	private void runIdentification() {

		/* Index properties:
		 1. Identity:
		 2. Exist:
		 3. Sequential Scattered:
		 4. Array:

		Table 1: supported formats by row and column indexes:
		 #  |  row      |  col     | Value |  example
		 --------------------------------------
		 1  | Identity  | Identity | Exist                   | csv, JSON/XML L                 single-line
		 2  | Identity  | Exist    | Exist                   | LibSVM                          single
		 3  | Identity  | Exist    | Not-Exist               | LibSVM+Pattern                  single
		 4  | Exist     | Exist    | Exist                   | MM Coordinate General           multi
		 5  | Array     | Array    | Exist                   | MM Array                        multi
		 6  | Exist     | Exist    | Partially-Exist         | MM Coordinate Symmetric         multi
		 7  | Exist     | Exist    | Partially-Exist+Pattern | MM Coordinate Skew-Symmetric    multi
		 8  | Exist     | Exist    | Not-Exist               | MM Coordinate Pattern           multi
		 9  | Exist     | Exist    | Not-Exist+Pattern       | MM Coordinate Symmetric Pattern multi
		 10 | SEQSCATTER| Identity | Exist                   | JSON/XML Multi Line, AMiner     multi

		strategy for checking the structure of indexes and values:
			1. map values:
				1.a values are full exist in the source
				1.b values are partially exist in the dataset (we have to check the Symmetric, Skew-Symmetric, and so on)
				1.c values are not exist in the source, in this case we have to check static value(s)
			2. map indexes:
				2.a after finding value properties the next step is looking for index maps, row index is in the first order
				2.b column index mapping
		 */

		// value mapping
		mapRow = mappingValues.getMapRow();
		mapCol = mappingValues.getMapCol();
		mapLen = mappingValues.getMapLen();
		mappingProperties = mappingValues.getMappingProperties();

		// save line by line index of string(index for Int, Long, float, Double, String, Boolean)
		sampleRawIndexes = mappingValues.getSampleRawIndexes();

		// matrix/frame properties for analysis and create datastructures
		nrows = mappingValues.getNrows();
		ncols = mappingValues.getNcols();
		nlines = mappingValues.getNlines();
		actualValueCount = mappingValues.getActualValueCount();
		staticColIndexes = new BitSet(ncols);

		// collect custom properties
		// 1. properties of row-index
		RowIndexStructure rowIndexStructure = getRowIndexStructure();

		// 2. properties of column-index
		ColIndexStructure colIndexStructure = getColIndexStructure();

		properties = new CustomProperties(mappingProperties, rowIndexStructure, colIndexStructure);
		properties.setNcols(ncols);

		// ref to Table 1:
		if(mappingProperties.getRecordProperties() == MappingProperties.RecordProperties.SINGLELINE) {
			// #1
			if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.Identity &&
				colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.Identity) {
				Pair<ArrayList<String>[], HashSet<String>[]> bckpsr = buildColsKeyPatternSingleRow();
				properties.setColKeyPatterns(bckpsr.getKey());
				properties.setEndWithValueStrings(bckpsr.getValue());
			}

			// #2
			else if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.Identity &&
				colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {
				// find cell-index and value separators
				RawIndex raw = null;
				for(int c = 0; c < ncols; c++) {
					if(mapCol[0][c] != -1) {
						raw = sampleRawIndexes[mapRow[0][c]];
						raw.cloneReservedPositions();
						break;
					}
				}
				HashMap<String, Long> indexDelimCount = new HashMap<>();
				String valueDelim = null;
				String indexDelim = null;
				Long maxCount = 0L;
				int begin = colIndexStructure.getColIndexBegin();
				for(int c = 0; c < ncols; c++) {
					if(mapCol[0][c] != -1) {
						Pair<Integer, Integer> pair = raw.findValue(c + begin);
						String tmpIndexDelim = raw.getSubString(pair.getKey() + pair.getValue(), mapCol[0][c]);
						if(indexDelimCount.containsKey(tmpIndexDelim))
							indexDelimCount.put(tmpIndexDelim, indexDelimCount.get(tmpIndexDelim) + 1);
						else
							indexDelimCount.put(tmpIndexDelim, 1L);
						if(maxCount < indexDelimCount.get(tmpIndexDelim)) {
							maxCount = indexDelimCount.get(tmpIndexDelim);
							indexDelim = tmpIndexDelim;
						}
						if(valueDelim == null) {
							int nextPos = raw.getNextNumericPosition(mapCol[0][c] + mapLen[0][c]);
							if(nextPos < raw.getRawLength()) {
								valueDelim = raw.getSubString(mapCol[0][c] + mapLen[0][c], nextPos);
							}
						}
					}
				}
				// update properties
				colIndexStructure.setIndexDelim(indexDelim);
				colIndexStructure.setValueDelim(valueDelim);
			}

		}
		else {
			// # 4, 6, 7, 8, 9
			if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.CellWiseExist &&
				colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {

				if(mappingProperties.getDataProperties() != MappingProperties.DataProperties.NOTEXIST) {
					Pair<ArrayList<String>, HashSet<String>> bvkpsr = buildValueKeyPattern();
					HashSet<String>[] endWithValueStrings = new HashSet[1];
					endWithValueStrings[0] = bvkpsr.getValue();
					properties.setValueKeyPattern(bvkpsr.getKey());
					properties.setEndWithValueStrings(endWithValueStrings);
				}

				int beginRowIndex = rowIndexStructure.getRowIndexBegin();
				int beginColIndex = colIndexStructure.getColIndexBegin();
				// build pattern for row-index
				Pair<ArrayList<String>, HashSet<String>> rowIndexPattern = buildIndexKeyPattern(true, beginRowIndex);
				rowIndexStructure.setKeyPattern(rowIndexPattern.getKey());
				rowIndexStructure.setEndWithValueString(rowIndexPattern.getValue());

				// build pattern for col-index
				Pair<ArrayList<String>, HashSet<String>> colIndexPattern = buildIndexKeyPattern(false, beginColIndex);
				colIndexStructure.setKeyPattern(colIndexPattern.getKey());
				colIndexStructure.setEndWithValueString(colIndexPattern.getValue());

			}
			// #10 sequential scattered
			if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.SeqScatter) {
				ArrayList<Pair<String, String>> prefixSuffixBeginEndCells = extractPrefixSuffixBeginEndCells(false);

				ArrayList<Pair<String, Set<Integer>>> keys;
				TextTrie textTrie = new TextTrie();
				textTrie.insert(prefixSuffixBeginEndCells.get(0).getKey(), 0);
				char startChar = prefixSuffixBeginEndCells.get(0).getKey().charAt(0);

				int minSubStringLength = Math.min(80, prefixSuffixBeginEndCells.get(0).getKey().length());
				for(int i = 1; i < prefixSuffixBeginEndCells.size(); i++) {
					String prefix = prefixSuffixBeginEndCells.get(i).getKey();
					for(int j = 0; j < prefix.length(); j++) {
						if(startChar == prefix.charAt(j))
							textTrie.insert(prefix.substring(j, j + Math.min(minSubStringLength, prefix.length() - j)),
								i);
					}
				}
				// scoring the prefix tree
				keys = textTrie.getAllKeys();
				String beginString = null;
				String endString = null;
				if(keys.get(0).getValue().size() == nrows) {
					int index = keys.get(0).getKey().indexOf("\n");
					if(index == -1)
						beginString = keys.get(0).getKey();
					else
						beginString = keys.get(0).getKey().substring(0, index);

					// recompute suffix strings to find end of string
					int minSuffixStringLength = prefixSuffixBeginEndCells.get(0).getValue().length();
					String reverseBeginString = new StringBuilder(beginString).reverse().toString();
					ArrayList<String> suffixes = new ArrayList<>();
					for(int i = 0; i < prefixSuffixBeginEndCells.size() - 1; i++) {
						String str = new StringBuilder(prefixSuffixBeginEndCells.get(i).getValue()).reverse()
							.toString();
						int indexBeginString = str.indexOf(reverseBeginString);
						if(indexBeginString != -1) {
							for(int j = indexBeginString + reverseBeginString.length(); j < str.length(); j++) {
								if(str.charAt(j) == '\n')
									indexBeginString++;
								else
									break;
							}
							minSuffixStringLength = Math.min(minSuffixStringLength, indexBeginString);
							suffixes.add(new StringBuilder(
								str.substring(0, indexBeginString + reverseBeginString.length())).reverse().toString());
						}
						else
							suffixes.add(str);
					}
					StringBuilder sbEndString = new StringBuilder();
					for(int i = 0; i < minSuffixStringLength; i++) {
						if(suffixes.get(0).length() == 0)
							break;
						char intersectChar = suffixes.get(0).charAt(i);
						if(intersectChar == '\n')
							break;
						boolean flag = true;
						for(String ss : suffixes) {
							if(ss.charAt(i) != intersectChar) {
								flag = false;
								break;
							}
						}
						if(flag)
							sbEndString.append(intersectChar);
						else
							break;
					}
					if(sbEndString.length() == 0)
						endString = beginString;
					else
						endString = sbEndString.toString();
					updateMapsAndExtractAllSuffixStringsOfColsMultiLine(beginString, endString);
					rowIndexStructure.setSeqBeginString(beginString);
					rowIndexStructure.setSeqEndString(endString);

					Pair<ArrayList<String>[], HashSet<String>[]> bckpsr = buildColsKeyPatternSingleRow();
					properties.setColKeyPatterns(bckpsr.getKey());
					properties.setEndWithValueStrings(bckpsr.getValue());
				}
				else {
					// TODO: extend sequential scattered format algorithm for heterogeneous structures
				}
			}
		}

		if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.CellWiseExist ||
			colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {
			properties.setSparse(true);
		}
	}

	// check row-index Exist
	// 1. row-index exist and can be reachable with a pattern
	// 2. row-index exist but there is no pattern for it
	// 3. row-index exist but just not for all cells! row-index appeared when the text broken newline="\n"
	private RowIndexStructure getRowIndexStructure() {
		// check the row index is a prefix string in sample raw, or the sample data line number equal to the sample matrix/frame row index
		// if the row indexes are in the prefix of values, so we need to build a key pattern to extract row indexes
		// to understanding row indexes are in sample raw we check just 3 column of data
		// to build a key pattern related to row indexes we just selected a row
		// TODO: decrease the number of row/col indexes want to check(3 or 5)

		RowIndexStructure rowIndexStructure = new RowIndexStructure();

		if(mappingProperties.getDataProperties() == MappingProperties.DataProperties.NOTEXIST) {
			if(nlines >= this.actualValueCount) {
				rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.CellWiseExist);
				rowIndexStructure.setRowIndexBegin(0);
				return rowIndexStructure;
			}
		}

		// check row-index Identity, the identity properties available just for
		// exist and partially exist mapped values
		if(mappingProperties.getDataProperties() != MappingProperties.DataProperties.NOTEXIST) {
			boolean identity = false;
			int missedCount = 0;

			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++)
					if(mapRow[r][c] != -1 && mapRow[r][c] != r) {
						missedCount++;
					}
			if((float) missedCount / actualValueCount < 0.07)
				identity = true;

			if(identity) {
				rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.Identity);
				return rowIndexStructure;
			}
		}

		BitSet[] bitSets = new BitSet[nrows];
		int[] rowCardinality = new int[nrows];
		int[] rowNZ = new int[nrows];
		boolean isCellWise = true;
		boolean isSeqScatter = true;
		boolean isExist = true;

		for(int r = 0; r < nrows; r++) {
			bitSets[r] = new BitSet(nlines);
			rowNZ[r] = 0;
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] != -1) {
					bitSets[r].set(mapRow[r][c]);
					rowNZ[r]++;
				}
			}
			rowCardinality[r] = bitSets[r].cardinality();
		}
		// check for Sequential:
		for(int r = 0; r < nrows && isSeqScatter; r++) {
			BitSet bitSet = bitSets[r];
			ArrayList<Integer> list = new ArrayList<>();
			for(int i = bitSet.nextSetBit(0); i != -1; i = bitSet.nextSetBit(i + 1))
				list.add(i);
			for(int i = 0; i < list.size() - 1 && isSeqScatter; i++)
				isSeqScatter = list.get(i) <= list.get(i + 1);
		}

		// check for Cell Wise
		for(int r = 0; r < nrows && isCellWise; r++)
			isCellWise = rowCardinality[r] == rowNZ[r];

		// check exist:
		int begin = 0;
		if(isCellWise) {
			for(int c = 0; c < ncols; c++) {
				begin = checkRowIndexesOnColumnRaw(c, 0);
				if(begin == -1) {
					isExist = false;
					break;
				}
			}
			if(isExist) {
				rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.CellWiseExist);
				rowIndexStructure.setRowIndexBegin(begin);
				return rowIndexStructure;
			}
		}
		else {
			ArrayList<RawIndex> list = new ArrayList<>();
			for(int r = 0; r < nrows; r++) {
				BitSet bitSet = bitSets[r];
				for(int i = bitSet.nextSetBit(0); i != -1; i = bitSet.nextSetBit(i + 1))
					list.add(sampleRawIndexes[i]);
				begin = checkRowIndexOnRaws(r, 0, list);
				if(begin == -1) {
					isExist = false;
					break;
				}
			}

			if(isExist) {
				rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.RowWiseExist);
				rowIndexStructure.setRowIndexBegin(begin);
				return rowIndexStructure;
			}

		}
		if(isSeqScatter) {
			rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.SeqScatter);
			return rowIndexStructure;
		}
		return rowIndexStructure;
	}
	private ColIndexStructure getColIndexStructure() {
		ColIndexStructure colIndexStructure = new ColIndexStructure();
		int begin = 0;
		boolean colIndexExist = true;

		if(mappingProperties.getDataProperties() == MappingProperties.DataProperties.NOTEXIST) {
			if(nlines >= this.actualValueCount) {
				colIndexStructure.setProperties(ColIndexStructure.IndexProperties.CellWiseExist);
				colIndexStructure.setColIndexBegin(0);
				return colIndexStructure;
			}
		}

		if(mappingProperties.getRecordProperties() == MappingProperties.RecordProperties.SINGLELINE) {
			// 1. check for column index are in the record
			for(int r = 0; r < Math.min(10, nrows); r++) {
				int rowIndex = -1;
				for(int c = 0; c < ncols; c++) {
					rowIndex = mapRow[r][c];
					if(rowIndex != -1)
						break;
				}
				begin = checkColIndexesOnRowRaw(rowIndex, 0);
				if(begin == -1) {
					colIndexExist = false;
					break;
				}
			}
			if(colIndexExist) {
				colIndexStructure.setColIndexBegin(begin);
				colIndexStructure.setProperties(ColIndexStructure.IndexProperties.CellWiseExist);
				return colIndexStructure;
			}
			// 2. check the column index are identity
			else {
				colIndexStructure.setProperties(ColIndexStructure.IndexProperties.Identity);
				return colIndexStructure;
			}
		}
		else {
			for(int r = 0; r < nrows && colIndexExist; r++) {
				for(int c = 0; c < Math.min(10, ncols) && colIndexExist; c++) {
					if(mapRow[r][c] != -1) {
						begin = checkColIndexOnRowRaw(mapRow[r][c], c, begin);
						colIndexExist = begin != -1;
					}
				}
			}

			if(colIndexExist) {
				colIndexStructure.setColIndexBegin(begin);
				colIndexStructure.setProperties(ColIndexStructure.IndexProperties.CellWiseExist);
				return colIndexStructure;
			}
		}

		return colIndexStructure;
	}
	private int checkRowIndexesOnColumnRaw(int colIndex, int beginPos) {
		int nne = 0;
		for(int r = 0; r < nrows; r++) {
			if(mapRow[r][colIndex] != -1) {
				RawIndex raw = sampleRawIndexes[mapRow[r][colIndex]];
				raw.cloneReservedPositions();
				Pair<Integer, Integer> pair = raw.findValue(r + beginPos);
				raw.restoreReservedPositions();
				if(pair == null)
					nne++;
			}
		}

		if(nne > 0) {
			if(beginPos == 1)
				return -1;
			else
				return checkRowIndexesOnColumnRaw(colIndex, 1);
		}
		else
			return beginPos;
	}
	private int checkRowIndexOnRaws(int rowIndex, int beginPos, ArrayList<RawIndex> list) {
		int nne = 0;
		for(RawIndex raw : list) {
			raw.cloneReservedPositions();
			Pair<Integer, Integer> pair = raw.findValue(rowIndex + beginPos);
			if(pair == null)
				nne++;
			raw.restoreReservedPositions();
		}

		if(nne > list.size() * 0.3) {
			if(beginPos == 1)
				return -1;
			else
				return checkRowIndexOnRaws(rowIndex, 1, list);
		}
		else
			return beginPos;
	}
	private int checkColIndexesOnRowRaw(int rowIndex, int beginPos) {
		int nne = 0;
		RawIndex raw = sampleRawIndexes[rowIndex];
		raw.cloneReservedPositions();
		for(int c = 0; c < ncols; c++) {
			if(mapCol[rowIndex][c] != -1) {
				Pair<Integer, Integer> pair = raw.findValue(c + beginPos);
				if(pair == null || pair.getKey() > mapCol[rowIndex][c])
					nne++;
			}
		}
		raw.restoreReservedPositions();
		if(nne > ncols * 0.05) {
			if(beginPos == 1)
				return -1;
			else
				return checkColIndexesOnRowRaw(rowIndex, 1);
		}
		else
			return beginPos;
	}
	private int checkColIndexOnRowRaw(int rowIndex, int colIndex, int beginPos) {
		RawIndex raw = sampleRawIndexes[rowIndex];
		raw.cloneReservedPositions();
		Pair<Integer, Integer> pair = raw.findValue(colIndex + beginPos);
		raw.restoreReservedPositions();

		if(pair == null) {
			if(beginPos == 1)
				return -1;
			else
				return checkColIndexOnRowRaw(rowIndex, colIndex, 1);
		}
		else
			return beginPos;
	}
	// Extract prefix strings:
	private ArrayList<Pair<String, String>> extractPrefixSuffixBeginEndCells(boolean reverse) {

		ArrayList<Pair<String, String>> result = new ArrayList<>();
		BitSet[] recordUsedLines = new BitSet[nlines];
		BitSet[] usedLines = new BitSet[nlines];
		for(int r = 0; r < nrows; r++)
			recordUsedLines[r] = new BitSet();

		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] != -1)
					recordUsedLines[r].set(mapRow[r][c]);

		for(int r = 0; r < nrows; r++) {
			usedLines[r] = new BitSet(nlines);
			for(int i = 0; i < nrows; i++) {
				if(i != r)
					usedLines[r].or(recordUsedLines[i]);
			}
		}
		int lastLine = 0;
		int lastPos = 0;
		for(int r = 0; r < nrows; r++) {
			int beginLine = 0;
			int endLine = 0;
			int beginPos = 0;
			int endPos;
			for(int i = 0; i < nlines; i++)
				if(recordUsedLines[r].get(i)) {
					beginLine = i;
					break;
				}
			for(int i = nlines - 1; i >= 0; i--)
				if(recordUsedLines[r].get(i)) {
					endLine = i;
					break;
				}

			endPos = 0;
			beginPos = sampleRawIndexes[beginLine].getRawLength();
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] == beginLine)
					beginPos = Math.min(beginPos, mapCol[r][c]);

				if(mapRow[r][c] == endLine)
					endPos = Math.max(endPos, mapCol[r][c] + mapLen[r][c]);
			}
			StringBuilder sbPrefix = new StringBuilder();
			if(lastLine != beginLine)
				sbPrefix.append(sampleRawIndexes[lastLine].getRaw().substring(lastPos)).append("\n");

			for(int i = lastLine + 1; i < beginLine; i++)
				sbPrefix.append(sampleRawIndexes[i].getRaw()).append("\n");
			sbPrefix.append(sampleRawIndexes[beginLine].getRaw().substring(0, beginPos));

			lastLine = endLine;
			lastPos = endPos;

			result.add(new Pair<>(sbPrefix.toString(), null));
		}

		// set suffix
		for(int r = 0; r < nrows - 1; r++) {
			result.get(r).setValue(result.get(r + 1).getKey());
		}
		result.get(nrows - 1).setValue(null);
		return result;
	}
	public CustomProperties getFormatProperties() {
		return properties;
	}
	private Pair<ArrayList<String>, HashSet<String>> buildValueKeyPattern() {
		int minSelectCols = Math.min(10, ncols);
		ArrayList<String>[] prefixesRemovedReverse = new ArrayList[1];
		ArrayList<String>[] prefixesRemoved = new ArrayList[1];
		ArrayList<String>[] prefixes = new ArrayList[1];
		ArrayList<String>[] suffixes = new ArrayList[1];
		ArrayList<Pair<String, Integer>>[] prefixesRemovedReverseSort = new ArrayList[1];
		ArrayList<String>[] keys = new ArrayList[minSelectCols];
		HashSet<String>[] colSuffixes = new HashSet[minSelectCols];
		LongestCommonSubsequence lcs = new LongestCommonSubsequence();

		for(int c = 0; c < minSelectCols; c++) {
			prefixesRemovedReverse[0] = new ArrayList<>();
			prefixes[0] = new ArrayList<>();
			suffixes[0]  = new ArrayList<>();
		}

		for(int c = 0; c < minSelectCols; c++) {
			prefixesRemovedReverse[0].addAll(extractAllPrefixStringsOfAColSingleLine(c, true, true).getKey());
			prefixes[0].addAll(extractAllPrefixStringsOfAColSingleLine(c,false, false).getKey());
			suffixes[0].addAll(extractAllSuffixStringsOfColsSingleLine(c, true));
		}
		HashSet<Integer> colIndexes = new HashSet<>();
		colIndexes.add(0);

		try {
			ExecutorService pool = CommonThreadPool.get(1);
			ArrayList<BuildColsKeyPatternSingleRowTask> tasks = new ArrayList<>();
			tasks.add(
				new BuildColsKeyPatternSingleRowTask(prefixesRemovedReverse, prefixesRemoved, prefixes, suffixes,
					prefixesRemovedReverseSort, keys, colSuffixes, lcs, colIndexes));

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();

			//check for exceptions
			for(Future<Object> task : rt)
				task.get();
		}
		catch(Exception e) {
			throw new RuntimeException("Failed BuildValueKeyPattern.", e);
		}

		return  new Pair<>(keys[0], colSuffixes[0]);
	}
	private String addToPrefixes(Set<String> list, String strValue, int value, boolean reverse){
		String str = reverse ? new StringBuilder(strValue).reverse().toString() : strValue;
		RawIndex rawIndex = new RawIndex(str);
		Pair<Integer, Integer> pair = rawIndex.findValue(value);
		if(pair != null) {
			String pstr = str.substring(0, pair.getKey());
			list.add(pstr);
			return Lop.OPERAND_DELIMITOR+str.substring(pair.getKey() + pair.getValue()).replaceAll("\\d", Lop.OPERAND_DELIMITOR);
		}
		return null;
	}
	private Pair<ArrayList<String>, HashSet<String>> buildIndexKeyPattern(boolean keyForRowIndexes, int begin) {
		ArrayList<String>[] prefixesRemovedReverse = new ArrayList[1];
		ArrayList<String>[] prefixesRemoved = new ArrayList[1];
		ArrayList<String>[] prefixes = new ArrayList[1];
		ArrayList<String>[] suffixes = new ArrayList[1];
		ArrayList<Pair<String, Integer>>[] prefixesRemovedReverseSort = new ArrayList[1];
		ArrayList<String>[] keys = new ArrayList[1];
		HashSet<String>[] colSuffixes = new HashSet[1];
		LongestCommonSubsequence lcs = new LongestCommonSubsequence();

		prefixesRemovedReverse[0] = new ArrayList<>();
		prefixesRemoved[0] = new ArrayList<>();
		prefixes[0] = new ArrayList<>();
		suffixes[0]  = new ArrayList<>();

		Map<Integer,ArrayList<Integer>> selectedRowColForIndexes = new HashMap<>();
		int maxSize = 0;
		for(int r = 1; r < nrows && maxSize < 1000; r++) {
			for(int c = 0; c < ncols && maxSize < 1000; c++) {
				if(mapCol[r][c] != -1 && r != c ) {
					selectedRowColForIndexes.computeIfAbsent(r, k -> new ArrayList<>());
					selectedRowColForIndexes.get(r).add(c);
					maxSize++;
				}
			}
		}
		if(keyForRowIndexes) {
			for(int r : selectedRowColForIndexes.keySet()) {
				ArrayList<Integer> colSet = selectedRowColForIndexes.get(r);
				ArrayList<String> tmpPrefixesRemovedReverse = extractAllPrefixStringsOfAColSingleLine(r, colSet, true, true).getKey();
				ArrayList<String> tmpPrefixesRemoved = extractAllPrefixStringsOfAColSingleLine(r, colSet, false, true).getKey();
				ArrayList<String> tmpPrefixes = extractAllPrefixStringsOfAColSingleLine(r, colSet, false, false).getKey();

				Set<String> tmpSet = new HashSet<>();
				for(String s : tmpPrefixesRemovedReverse) {
					String suf = addToPrefixes(tmpSet, s, r+begin, true);
					if(suf != null)
						suffixes[0].add(suf);
				}
				prefixesRemovedReverse[0].addAll(tmpSet);

				tmpSet = new HashSet<>();
				for(String s : tmpPrefixesRemoved)
					addToPrefixes(tmpSet, s, r+begin, false);
				prefixesRemoved[0].addAll(tmpSet);

				tmpSet = new HashSet<>();
				for(String s : tmpPrefixes)
					addToPrefixes(tmpSet, s, r+begin, false);
				prefixes[0].addAll(tmpSet);
			}
		}
		else {
			for(int r : selectedRowColForIndexes.keySet()) {
				ArrayList<Integer> colSet = selectedRowColForIndexes.get(r);
				ArrayList<String> tmpPrefixesRemovedReverse = extractAllPrefixStringsOfAColSingleLine(r, colSet, true, true).getKey();
				ArrayList<String> tmpPrefixesRemoved = extractAllPrefixStringsOfAColSingleLine(r, colSet, false, true).getKey();
				ArrayList<String> tmpPrefixes = extractAllPrefixStringsOfAColSingleLine(r, colSet, false, false).getKey();

				Set<String> tmpSet = new HashSet<>();
				for(String s : tmpPrefixesRemovedReverse) {
					for(int c: colSet) {
						String suf = addToPrefixes(tmpSet, s, c+begin, true);
						if(suf != null)
							suffixes[0].add(suf);
					}
				}
				prefixesRemovedReverse[0].addAll(tmpSet);

				tmpSet = new HashSet<>();
				for(String s : tmpPrefixesRemoved)
					for(int c: colSet) {
						addToPrefixes(tmpSet, s, c+begin, false);
					}
				prefixesRemoved[0].addAll(tmpSet);

				tmpSet = new HashSet<>();
				for(String s : tmpPrefixes)
					for(int c: colSet) {
						addToPrefixes(tmpSet, s, c+begin, false);
					}
				prefixes[0].addAll(tmpSet);
			}
		}

		HashSet<Integer> colIndexe = new HashSet<>();
		colIndexe.add(0);

		try {
			ExecutorService pool = CommonThreadPool.get(1);
			ArrayList<BuildColsKeyPatternSingleRowTask> tasks = new ArrayList<>();
			tasks.add(
				new BuildColsKeyPatternSingleRowTask(prefixesRemovedReverse, prefixesRemoved, prefixes, suffixes,
					prefixesRemovedReverseSort, keys, colSuffixes, lcs, colIndexe));

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();

			//check for exceptions
			for(Future<Object> task : rt)
				task.get();
		}
		catch(Exception e) {
			throw new RuntimeException("Failed BuildValueKeyPattern.", e);
		}

		return  new Pair<>(keys[0], colSuffixes[0]);
	}

	// Get all prefix strings of a column
	public Pair<ArrayList<String>[], ArrayList<Integer>[]> extractAllPrefixStringsOfColsSingleLine(boolean reverse, boolean removesSelected) {
		ArrayList<String>[] prefixStrings = new ArrayList[ncols];
		ArrayList<Integer>[] rowIndexes = new ArrayList[ncols];
		for(int c = 0; c < ncols; c++) {
			Pair<ArrayList<String>, ArrayList<Integer>> pair = extractAllPrefixStringsOfAColSingleLine(c, reverse, removesSelected);
			prefixStrings[c] = pair.getKey();
			rowIndexes[c] = pair.getValue();
		}
		return new Pair<>(prefixStrings, rowIndexes);
	}
	public Pair<ArrayList<String>, ArrayList<Integer>> extractAllPrefixStringsOfAColSingleLine(int r,
		ArrayList<Integer> colIndexes, boolean reverse, boolean removesSelected) {
		ArrayList<String> prefixStrings = new ArrayList();
		ArrayList<Integer> rowIndexes = new ArrayList();
		for(int c : colIndexes) {
			int rowIndex = mapRow[r][c];
			if(rowIndex != -1) {
				rowIndexes.add(rowIndex);
				String str;
				if(removesSelected)
					str = sampleRawIndexes[rowIndex].getRemainedTexts(0, mapCol[r][c]);
				else
					str = sampleRawIndexes[rowIndex].getRaw().substring(0, mapCol[r][c]);
				if(reverse)
					prefixStrings.add(new StringBuilder(str).reverse().toString());
				else
					prefixStrings.add(str);
			}
		}
		return new Pair<>(prefixStrings, rowIndexes);
	}

	public Pair<ArrayList<String>, ArrayList<Integer>> extractAllPrefixStringsOfAColSingleLine(int colIndex,
		boolean reverse, boolean removesSelected) {
		ArrayList<String> prefixStrings = new ArrayList();
		ArrayList<Integer> rowIndexes = new ArrayList();
		for(int r = 0; r < nrows; r++) {
			int rowIndex = mapRow[r][colIndex];
			if(rowIndex != -1) {
				rowIndexes.add(rowIndex);
				String str;
				if(removesSelected)
					str = sampleRawIndexes[rowIndex].getRemainedTexts(0, mapCol[r][colIndex]);
				else
					str = sampleRawIndexes[rowIndex].getRaw().substring(0, mapCol[r][colIndex]);
				if(reverse)
					prefixStrings.add(new StringBuilder(str).reverse().toString());
				else
					prefixStrings.add(str);
			}
		}
		return new Pair<>(prefixStrings, rowIndexes);
	}


	private ArrayList<String>[] extractAllSuffixStringsOfColsSingleLine(boolean removeData) {
		ArrayList<String>[] result = new ArrayList[ncols];
		for(int c = 0; c < ncols; c++) {
			result[c] = new ArrayList<>();
			for(int r = 0; r < nrows; r++) {
				int rowIndex = mapRow[r][c];
				if(rowIndex == -1)
					continue;
				String str;
				if(removeData)
					str = sampleRawIndexes[rowIndex].getRemainedTexts(mapCol[r][c] + mapLen[r][c], -1);
				else
					str = sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][c] + mapLen[r][c]);
				result[c].add(str);
			}
		}
		return result;
	}

	private ArrayList<String> extractAllSuffixStringsOfColsSingleLine(int col, boolean removeData) {
		ArrayList<String> result = new ArrayList<>();
		for(int r = 0; r < nrows; r++) {
			int rowIndex = mapRow[r][col];
			if(rowIndex == -1)
				continue;
			String str;
			if(removeData)
				str = sampleRawIndexes[rowIndex].getRemainedTexts(mapCol[r][col] + mapLen[r][col], -1);
			else
				str = sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][col] + mapLen[r][col]);
			result.add(str);
		}
		return result;
	}

	private ArrayList<String> extractAllSuffixStringsOfColsSingleLine(ArrayList<Integer> rows,int col, boolean removeData) {
		ArrayList<String> result = new ArrayList<>();
		for(int r: rows) {
			int rowIndex = mapRow[r][col];
			if(rowIndex == -1)
				continue;
			String str;
			if(removeData)
				str = sampleRawIndexes[rowIndex].getRemainedTexts(mapCol[r][col] + mapLen[r][col], -1);
			else
				str = sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][col] + mapLen[r][col]);
			result.add(str);
		}
		return result;
	}

	private void updateMapsAndExtractAllSuffixStringsOfColsMultiLine(String beginString, String endString) {
		RawIndex[] upRawIndexes = new RawIndex[nrows];
		ArrayList<Pair<Integer, Integer>> beginIndexes = getTokenIndexOnMultiLineRecords(beginString);
		ArrayList<Pair<Integer, Integer>> endIndexes;
		String endToken;
		if(!beginString.equals(endString)) {
			endIndexes = getTokenIndexOnMultiLineRecords(endString);
			endToken = endString;
		}
		else {
			endIndexes = new ArrayList<>();
			for(int i = 1; i < beginIndexes.size(); i++)
				endIndexes.add(beginIndexes.get(i));
			endIndexes.add(new Pair<>(this.sampleRawIndexes.length - 1,
				this.sampleRawIndexes[this.sampleRawIndexes.length - 1].getRawLength()));
			endToken = "";
		}
		int r = 0;
		int i = 0;
		int j = 0;
		StringBuilder sb = new StringBuilder();
		while(i < beginIndexes.size() && j < endIndexes.size() && r < nrows) {
			Pair<Integer, Integer> p1 = beginIndexes.get(i);
			Pair<Integer, Integer> p2 = endIndexes.get(j);
			int n = 0;
			while(p1.getKey() < p2.getKey() || (p1.getKey() == p2.getKey() && p1.getValue() < p2.getValue())) {
				n++;
				i++;
				if(i == beginIndexes.size())
					break;
				p1 = beginIndexes.get(i);
			}
			j += n - 1;
			sb.append(this.sampleRawIndexes[beginIndexes.get(i - n).getKey()].getRaw()
				.substring(beginIndexes.get(i - n).getValue()));
			for(int ri = beginIndexes.get(i - n).getKey() + 1; ri < endIndexes.get(j).getKey(); ri++) {
				sb.append(this.sampleRawIndexes[ri].getRaw());
			}
			sb.append(this.sampleRawIndexes[endIndexes.get(j).getKey()].getRaw()
				.substring(0, endIndexes.get(j).getValue())).append(endToken);
			RawIndex rawIndex = new RawIndex();
			rawIndex.setRaw(sb.toString());
			sb = new StringBuilder();
			j++;
			// update mapping
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] != -1) {
					if(mapRow[r][c] != beginIndexes.get(i - n).getKey())
						this.mapCol[r][c] +=
							this.sampleRawIndexes[beginIndexes.get(i - n).getKey()].getRawLength() -
								beginIndexes.get(i - n).getValue();
					else
						this.mapCol[r][c] -= beginIndexes.get(i - n).getValue();

					for(int ci = beginIndexes.get(i - n).getKey() + 1; ci < this.mapRow[r][c]; ci++)
						this.mapCol[r][c] += this.sampleRawIndexes[ci].getRawLength();
					rawIndex.setReservedPositions(mapCol[r][c], mapLen[r][c]);
					this.mapRow[r][c] = r;
				}
			}
			upRawIndexes[r] = rawIndex;
			r++;
		}
		this.sampleRawIndexes = upRawIndexes;
	}

	private ArrayList<Pair<Integer, Integer>> getTokenIndexOnMultiLineRecords(String token) {
		ArrayList<Pair<Integer, Integer>> result = new ArrayList<>();

		for(int ri = 0; ri < this.sampleRawIndexes.length; ri++) {
			String raw = this.sampleRawIndexes[ri].getRaw();
			int index;
			int fromIndex = 0;
			do {
				index = raw.indexOf(token, fromIndex);
				if(index != -1) {
					result.add(new Pair<>(ri, index));
					fromIndex = index + token.length();
				}
				else
					break;
			}
			while(true);
		}
		return result;
	}

	private ArrayList<Pair<Integer, Integer>> getTokenIndexOnMultiLineRecords(String beginToken, String endToken) {
		ArrayList<Pair<Integer, Integer>> result = new ArrayList<>();

		for(int ri = 0; ri < this.sampleRawIndexes.length; ) {
			String raw = this.sampleRawIndexes[ri].getRaw();
			int index;
			int fromIndex = 0;
			do {
				index = raw.indexOf(endToken, fromIndex);
				if(index != -1) {
					if(index + endToken.length() + beginToken.length() <= raw.length()) {
						boolean flag = true;
						for(int i = index + endToken.length(), j = 0;
							i < index + endToken.length() + beginToken.length() && flag;
							i++, j++) {
							flag = raw.charAt(i) == beginToken.charAt(j);
						}
						if(flag) {
							result.add(new Pair<>(ri, index));
							fromIndex = index + beginToken.length() + endToken.length();
						}
						else
							fromIndex++;
					}
					else {
						if(ri + 1 == this.sampleRawIndexes.length)
							break;
						// skip empty rows
						do {
							raw = this.sampleRawIndexes[++ri].getRaw();
						}
						while(raw.length() == 0);

						if(raw.startsWith(beginToken)) {
							result.add(new Pair<>(ri, 0));
							fromIndex = 1;
						}
					}
				}
				else
					break;
			}
			while(true);
			ri++;
		}
		return result;
	}

	private Pair<Set<String>, Set<String>> getNewRefineKeys(LongestCommonSubsequence lcs, String firstKey,
		ArrayList<String> prefixesRemoved, ArrayList<String> prefixes, Set<String> refineKeys) {

		Set<String> setRefineLCS = new HashSet<String>();
		Set<String> newSetRefineLCS = new HashSet<String>();

		for(String refineKey : refineKeys) {
			boolean flagRefine = true;
			boolean isInTheMiddleOfString = false;
			String[] lcsKey = (refineKey+Lop.OPERAND_DELIMITOR+firstKey).split(Lop.OPERAND_DELIMITOR);
			ArrayList<String> tmpList = new ArrayList<>();
			for(String sk : lcsKey)
				if(sk.length() > 0)
					tmpList.add(sk);

			for(int i = 0; i < prefixes.size() && !isInTheMiddleOfString; i++) {
				String str = prefixes.get(i);
				int indexOnString = getIndexOfKeyPatternOnString(str, tmpList, 0);
				flagRefine &= indexOnString == str.length();
				if(!flagRefine)
					isInTheMiddleOfString = indexOnString != -1;
			}
			if(flagRefine)
				setRefineLCS.add(refineKey);
			else if(!isInTheMiddleOfString) {
				for(int i = 0; i < prefixesRemoved.size() ; i++) {
					String psStr = prefixesRemoved.get(i).substring(0, firstKey.length());
					ArrayList<String> list1 = lcs.getLCS(refineKey, psStr);
					Set<String> set = new HashSet<String>();
					set.addAll(list1);

					for(String lcsKeys : set) {
						if(setRefineLCS.contains(lcsKeys) || newSetRefineLCS.contains(lcsKey))
							continue;
						String[] newLCSKey = (lcsKeys+Lop.OPERAND_DELIMITOR+firstKey).split(Lop.OPERAND_DELIMITOR);
						ArrayList<String> tmpLCSKeyList = new ArrayList<>();
						for(String sk : newLCSKey)
							if(sk.length() > 0)
								tmpLCSKeyList.add(sk);

						boolean str1Check = getIndexOfKeyPatternOnString(psStr + firstKey, tmpLCSKeyList, 0) == 	psStr.length();
						if(str1Check)
							newSetRefineLCS.add(lcsKeys);
					}
				}
			}
		}
		return new Pair<>(setRefineLCS, newSetRefineLCS);
	}

	private Set<String> getRefineKeysStep(LongestCommonSubsequence lcs, String string1, String string2,
		String psString1, String psString2, String firstKey){
		// remove first key from end of Str1 and Str2
		String str1 = string1.substring(0, string1.length() - firstKey.length());
		String str2 = string2.substring(0, string2.length() - firstKey.length());

		ArrayList<String> list1 = lcs.getLCS(str1, str2);
		Set<String> setLCS = new HashSet<String>();
		setLCS.addAll(list1);

		Set<String> refineKeysStep = new HashSet<>();
		for(String lcsKeys : setLCS) {
			String[] lcsKey = (lcsKeys+Lop.OPERAND_DELIMITOR+firstKey).split(Lop.OPERAND_DELIMITOR);
			ArrayList<String> tmpList = new ArrayList<>();
			for(String sk : lcsKey)
				if(sk.length() > 0)
					tmpList.add(sk);

			boolean str1Check = getIndexOfKeyPatternOnString(psString1, tmpList, 0) == psString1.length();
			boolean str2Check = getIndexOfKeyPatternOnString(psString2, tmpList, 0) == psString2.length();
			if(str1Check && str2Check)
				refineKeysStep.add(lcsKeys);
		}
		return refineKeysStep;
	}

	private ArrayList<String> cleanUPKey(ArrayList<String> keys, ArrayList<String> prefixes){
		ArrayList<String> result = new ArrayList<>();
		int i = keys.size() -1;
		for(; i>=0; i--) {
			boolean flag = true;
			for(int j =0; j< prefixes.size() && flag; j++) {
				String bk = keys.get(i);
				int k1 = getIndexOfKeyPatternOnString(prefixes.get(j), i, keys, 0);
				int k2 = prefixes.get(j).length();
				flag = getIndexOfKeyPatternOnString(prefixes.get(j), i, keys, 0) == prefixes.get(j).length();
			}
			if(flag)
				break;
		}
		if( i == -1)
			return keys;
		else {
			for(int index = i; index< keys.size(); index++)
				result.add(keys.get(index));

			int a = 100;
		}
		return result;
	}
	private boolean checkExtraKeyForCol(ArrayList<String> keys, String extraKey , ArrayList<String> prefixes){
		boolean flag = true;
		for(int i=0; i<keys.size()-1 && flag; i++)
			flag = keys.get(i).equals(keys.get(i+1));
		if(!flag)
			return false;
		for(int j = 0; j < prefixes.size() && flag; j++) {
			int index = prefixes.get(j).indexOf(extraKey);
			if(index !=-1) {
				index += extraKey.length();
				flag = getIndexOfKeyPatternOnString(prefixes.get(j), 0, keys, index) == prefixes.get(j).length();
			}
			else
				flag = false;
		}
		return flag;
	}
	private Integer getIndexOfKeyPatternOnString(String str, ArrayList<String> key, int beginPos) {
		return getIndexOfKeyPatternOnString(str,0, key, beginPos);
	}
	private Integer getIndexOfKeyPatternOnString(String str, int keyFromIndex,ArrayList<String> key, int beginPos) {
		int currPos = beginPos;
		boolean flag = true;
		for(int i = keyFromIndex; i < key.size(); i++) {
			int index = str.indexOf(key.get(i), currPos);
			if(index != -1)
				currPos = index + key.get(i).length();
			else {
				flag = false;
				break;
			}
		}
		if(flag)
			return currPos;
		else
			return -1;
	}
	private Pair<ArrayList<String>[], HashSet<String>[]> buildColsKeyPatternSingleRow() {
		ArrayList<String>[] prefixesRemovedReverse = extractAllPrefixStringsOfColsSingleLine(true, true).getKey();
		ArrayList<String>[] prefixesRemoved = new ArrayList[ncols];
		ArrayList<String>[] prefixes = extractAllPrefixStringsOfColsSingleLine(false, false).getKey();
		ArrayList<String>[] suffixes = extractAllSuffixStringsOfColsSingleLine(true);
		ArrayList<Pair<String, Integer>>[] prefixesRemovedReverseSort = new ArrayList[ncols];
		ArrayList<String>[] keys = new ArrayList[ncols];
		HashSet<String>[] colSuffixes = new HashSet[ncols];
		LongestCommonSubsequence lcs = new LongestCommonSubsequence();

		int numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		try {
			ExecutorService pool = CommonThreadPool.get(numThreads);
			ArrayList<BuildColsKeyPatternSingleRowTask> tasks = new ArrayList<>();
			int blklen = (int) Math.ceil((double) ncols / (numThreads * numThreads));
			for(int i = 0; i < numThreads; i++) {
				HashSet<Integer> colIndexes = new HashSet<>();
				for(int j = 0; j < numThreads && j * numThreads * blklen + i * blklen < ncols; j++) {
					int begin = j * numThreads * blklen + i * blklen;
					int end = Math.min(j * numThreads * blklen + (i + 1) * blklen, ncols);
					for(int k = begin; k < end; k++)
						colIndexes.add(k);
				}
				tasks.add(
					new BuildColsKeyPatternSingleRowTask(prefixesRemovedReverse, prefixesRemoved, prefixes, suffixes,
						prefixesRemovedReverseSort, keys, colSuffixes, lcs, colIndexes));
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();

			//check for exceptions
			for(Future<Object> task : rt)
				task.get();

			int a = 50;
		}
		catch(Exception e) {
			throw new RuntimeException("Failed parallel ColsKeyPatternSingleRow.", e);
		}
		return  new Pair<>(keys, colSuffixes);
	}
	private class BuildColsKeyPatternSingleRowTask implements Callable<Object> {
		private final ArrayList<String>[] prefixesRemovedReverse;
		private final ArrayList<String>[] prefixesRemoved;
		private final ArrayList<String>[] prefixes;
		private final ArrayList<String>[] suffixes;
		private final ArrayList<Pair<String, Integer>>[] prefixesRemovedReverseSort;
		private final ArrayList<String>[] keys;
		private final HashSet<String>[] colSuffixes;
		private final LongestCommonSubsequence lcs;
		private final HashSet<Integer> colIndexes;

		public BuildColsKeyPatternSingleRowTask(ArrayList<String>[] prefixesRemovedReverse,
			ArrayList<String>[] prefixesRemoved, ArrayList<String>[] prefixes, ArrayList<String>[] suffixes,
			ArrayList<Pair<String, Integer>>[] prefixesRemovedReverseSort, ArrayList<String>[] keys,
			HashSet<String>[] colSuffixes, LongestCommonSubsequence lcs, HashSet<Integer> colIndexes) {
			this.prefixesRemovedReverse = prefixesRemovedReverse;
			this.prefixesRemoved = prefixesRemoved;
			this.prefixes = prefixes;
			this.suffixes = suffixes;
			this.prefixesRemovedReverseSort = prefixesRemovedReverseSort;
			this.keys = keys;
			this.colSuffixes = colSuffixes;
			this.lcs = lcs;
			this.colIndexes = colIndexes;
		}
		@Override
		public Object call() throws Exception {
			// Sort prefixesRemovedReverse list
			for(int c :colIndexes){
				keys[c] = new ArrayList<>();
				Map<String, ArrayList<Integer>> mapPrefixesRemovedReverse = new HashMap<>();
				for(int i=0; i<prefixesRemovedReverse[c].size(); i++) {
					StringBuilder sb = new StringBuilder();
					String str = prefixesRemovedReverse[c].get(i).replaceAll("\\d", Lop.OPERAND_DELIMITOR);
					for(int j = 0; j< str.length(); j++){
						String charStr = str.charAt(j)+"";
						if(!charStr.equals(Lop.OPERAND_DELIMITOR))
							sb.append(charStr);
						else if(sb.length() == 0 || !(sb.charAt(sb.length() -1)+"").equals(Lop.OPERAND_DELIMITOR))
							sb.append(Lop.OPERAND_DELIMITOR);
					}
					String sbStr = sb.toString();
					if(!mapPrefixesRemovedReverse.containsKey(sbStr))
						mapPrefixesRemovedReverse.put(sbStr, new ArrayList<>());
					mapPrefixesRemovedReverse.get(sbStr).add(i);
				}
				prefixesRemovedReverse[c] = new ArrayList<>();
				prefixesRemoved[c] = new ArrayList<>();
				prefixesRemovedReverseSort[c] = new ArrayList<>();

				for(String s: mapPrefixesRemovedReverse.keySet()){
					prefixesRemovedReverseSort[c].add(new Pair<>(s, mapPrefixesRemovedReverse.get(s).get(0)));
				}
				prefixesRemovedReverseSort[c].sort(AscendingPairStringComparator);
				for(Pair<String, Integer> pair: prefixesRemovedReverseSort[c]){
					prefixesRemovedReverse[c].add(pair.getKey());
					prefixesRemoved[c].add(new StringBuilder(pair.getKey()).reverse().toString());
				}
			}

			// build patterns:
			for(int c :colIndexes) {
				if(prefixesRemoved[c].size() == 1){
					keys[c] = new ArrayList<>();
					if(prefixesRemoved[c].get(0).length() == 0 || prefixesRemoved[c].get(0).equals(Lop.OPERAND_DELIMITOR))
						keys[c].add("");

					String[] lcsKey = prefixesRemoved[c].get(0).split(Lop.OPERAND_DELIMITOR);
					for(String sk : lcsKey)
						if(sk.length() > 0)
							keys[c].add(sk);
					continue;
				}

				String firstKey;
				// STEP 1: find fist key:
				String selectedString = prefixesRemoved[c].get(0);
				boolean flag = true;
				StringBuilder sbToken = new StringBuilder();
				sbToken.append(selectedString.charAt(selectedString.length() -1));
				for(int i = 2; i < selectedString.length() && flag; i++) {
					char ch = selectedString.charAt(selectedString.length()-i);
					for(int j = 1; j < prefixesRemoved[c].size() && flag; j++) {
						String str = prefixesRemoved[c].get(j);
						flag = str.charAt(str.length()-i) == ch;
					}
					if(flag)
						sbToken.append(ch);
				}
				firstKey = sbToken.reverse().toString();
				flag = true;

				String[] lcsKey = firstKey.split(Lop.OPERAND_DELIMITOR);
				ArrayList<String> tmpList = new ArrayList<>();
				for(String sk : lcsKey)
					if(sk.length() > 0)
						tmpList.add(sk);

				for(int i = 0; i < prefixes[c].size() && flag; i++)
					flag = getIndexOfKeyPatternOnString(prefixes[c].get(i), tmpList, 0) == prefixes[c].get(i).length();

				if(flag) {
					keys[c] = tmpList;
					continue;
				}
				// STEP 2: add another keys
				int indexI = 0;
				int indexJ = 0;
				Set<String> refineKeysStep = new HashSet<>();
				do {
					for(; indexI < prefixesRemovedReverseSort[c].size() - 1 && refineKeysStep.size() == 0; indexI++) {
						String str1 = prefixesRemoved[c].get(indexI);
						String psStr1 = prefixes[c].get(prefixesRemovedReverseSort[c].get(indexI).getValue());
						for(indexJ = indexI + 1;
							indexJ < prefixesRemovedReverseSort[c].size() && refineKeysStep.size() == 0;
							indexJ++) {
							String str2 = prefixesRemoved[c].get(indexJ);
							String psStr2 = prefixes[c].get(prefixesRemovedReverseSort[c].get(indexJ).getValue());
							refineKeysStep = getRefineKeysStep(lcs, str1, str2, psStr1, psStr2, firstKey);
						}
					}
					if(indexI < prefixesRemovedReverse[c].size() -1 && indexJ < prefixesRemovedReverse[c].size())
						break;

					do {
						Pair<Set<String>, Set<String>> pair = getNewRefineKeys(lcs, firstKey, prefixesRemoved[c], prefixes[c], refineKeysStep);
						refineKeysStep = pair.getKey();
						if(pair.getValue().size() == 0)
							break;
						else
							refineKeysStep.addAll(pair.getValue());
					}
					while(true);

				} while(refineKeysStep.size() == 0);

				if(refineKeysStep.size() == 0) {
					// TODO: we have to apply tokenizer
				}
				else if(refineKeysStep.size() == 1) {
					String[] refinedLCSKey = (refineKeysStep.iterator().next()+Lop.OPERAND_DELIMITOR+firstKey).split(Lop.OPERAND_DELIMITOR);
					keys[c] = new ArrayList<>();
					for(String sk : refinedLCSKey)
						if(sk.length() > 0)
							keys[c].add(sk);
				}
				else{
					ArrayList<String> sortedStrings = new ArrayList<>();
					sortedStrings.addAll(refineKeysStep);
					Collections.sort(sortedStrings, AscendingStringLengthComparator);
					String[] refinedLCSKey = (sortedStrings.get(sortedStrings.size()-1)+Lop.OPERAND_DELIMITOR+firstKey).split(Lop.OPERAND_DELIMITOR);
					keys[c] = new ArrayList<>();
					for(String sk : refinedLCSKey)
						if(sk.length() > 0)
							keys[c].add(sk);
				}
			}

			// CleanUP keys: reduce key list if it possible
			for(int c :colIndexes) {
				if(c == 5){
					int fff = 500;
				}
				ArrayList<String> cleanUPKeys =  cleanUPKey(keys[c], prefixes[c]);
//				boolean flagOptimal = false;
//				for(int i=0; i< keys[c].size() && !flagOptimal; i++)
//					flagOptimal = keys[c].get(i).contains(" ");
//				if(flagOptimal) {
//					keys[c] = optimalKeyPattern(keys[c], prefixes[c]);
//				}

				// set static col flag
				Boolean flagFixCol = true;
				for(int r = 0; r < nrows && flagFixCol && prefixes[c].size() !=nrows; r++){
					String rawStr =  sampleRawIndexes[r].getRaw();
					flagFixCol = getIndexOfKeyPatternOnString(rawStr, cleanUPKeys, 0) !=-1;
				}
				staticColIndexes.set(c, flagFixCol);
				if(!flagFixCol && cleanUPKeys.size() < keys[c].size()){
					String extraKey = keys[c].get(keys[c].size()-cleanUPKeys.size()-1);
					if(checkExtraKeyForCol(cleanUPKeys, extraKey,prefixes[c])){
						keys[c] = new ArrayList<>();
						keys[c].add(extraKey);
						keys[c].addAll(cleanUPKeys);
					}
					else
						keys[c] = cleanUPKeys;
				}
				else
					keys[c] = cleanUPKeys;

				// Build suffixes
				Set<String> setSuffix = new HashSet<>();
				TextTrie suffixTrie = new TextTrie();
				for(String su: suffixes[c]) {
					String[] suffixesList = su.split(Lop.OPERAND_DELIMITOR, -1);
					if(suffixesList.length > 0) {
						if(suffixesList.length == 1 && suffixesList[0].length() == 0)
							continue;
						if(suffixesList[1].length() < suffixStringLength)
							setSuffix.add(suffixesList[1]);
						else
							setSuffix.add(suffixesList[1].substring(0, suffixStringLength));
					}
				}
				if(setSuffix.size() == 0) {
					colSuffixes[c] = new HashSet<>();
					continue;
				}
				int rowIndexSuffix = 0;
				for(String ss: setSuffix){
					suffixTrie.insert(ss, rowIndexSuffix++);
				}
				HashSet<String> colSuffixe = new HashSet<>();
				ArrayList<Pair<String, Set<Integer>>> allSuffixes = suffixTrie.getAllKeys();
				if(allSuffixes.get(0).getValue().size() == setSuffix.size())
					colSuffixe.add(allSuffixes.get(0).getKey());
				else {
					Set<Integer> coveredRowIndexes = new HashSet<>();
					for(Pair<String, Set<Integer>> p: allSuffixes){
						int currentSize = coveredRowIndexes.size();
						coveredRowIndexes.addAll(p.getValue());
						if(currentSize != coveredRowIndexes.size())
							colSuffixe.add(p.getKey());
					}
				}
				colSuffixes[c] = colSuffixe;
			}
			return new Pair<>(keys, colSuffixes);
		}
	}
	public String getConflictToken(int[] cols) {
		boolean flagStatic = true;
		for(int c=0; c<cols.length && flagStatic ; c++){
			flagStatic = staticColIndexes.get(cols[c]);
		}
		if(flagStatic)
			return null;

		int lastColIndex = cols[cols.length - 1];
		ArrayList<String> suffixesBetweenBeginEnd = new ArrayList<>();
		ArrayList<String> suffixesRefine = extractAllSuffixStringsOfColsSingleLine(lastColIndex, true);
		Set<String> setSuffixesRefine = new HashSet<>();
		setSuffixesRefine.addAll(suffixesRefine);
		if(setSuffixesRefine.size() == 1 && setSuffixesRefine.iterator().next().length() == 0)
			return null;

		int rowIndex;
		for(int r = 0; r < nrows; r++) {
			ArrayList<Integer> filledCols = new ArrayList<>();
			for(int c: cols){
				if(mapCol[r][c] !=-1)
					filledCols.add(c);
			}
			if(filledCols.size() <= 1  || (rowIndex=mapRow[r][filledCols.get(0)]) == -1)
				continue;

			int ib = filledCols.get(0);
			int ie = filledCols.get(filledCols.size() -1);
			String str = sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][ib] + mapLen[r][ib], mapCol[r][ie]);
			suffixesBetweenBeginEnd.add(str);
		}

		ArrayList<String> containList = new ArrayList<>();
		int maxTokenLength = 0;
		String selectedString = "";
		for(String suf : suffixesRefine) {
			int index = suf.indexOf(Lop.OPERAND_DELIMITOR, 1);
			if(index == -1)
				index = suf.length();
			String str;
			if((str = suf.substring(1, index)).length() > 0) {
				containList.add(str);
				if(maxTokenLength == 0 || maxTokenLength > str.length()) {
					maxTokenLength = str.length();
					selectedString = str;
				}
			}
		}
		if(containList.size() == 0)
			return null;

		Map<Integer, ArrayList<String>> conflicts = new HashMap<>();
		maxTokenLength = Math.min(maxTokenLength, 50);
		for(int tl = 1; tl < maxTokenLength; tl++) {
			ArrayList<String> tokens = stringTokenize(selectedString, tl);
			conflicts.put(tl, new ArrayList<>());
			for(String t : tokens) {
				boolean flag = false;
				for(String between : suffixesBetweenBeginEnd) {
					flag = between.contains(t);
					if(flag)
						break;
				}
				if(!flag)
					conflicts.get(tl).add(t);
			}
		}
		ArrayList<Pair<String, ArrayList<Integer>>> candidate = new ArrayList<>();
		ArrayList<String> tokens = new ArrayList<>();
		for(int i = maxTokenLength - 1; i > 0 ; i--) {
			for(String tc : conflicts.get(i)) {
				boolean flag = false;
				for(String currenToken: tokens)
					if(currenToken.startsWith(tc)){
						flag = true;
						break;
					}
				if(flag)
					continue;
				else flag = true;
				ArrayList<Integer> distances = new ArrayList<>();
				boolean containZero = false;
				for(String s : containList) {
					int index = s.indexOf(tc);
					flag = index!=-1;
					if(!flag)
						break;
					else {
						distances.add(index);
						containZero |= index == 0;
					}
				}
				if(flag) {
					if(containZero)
						return tc;
					candidate.add(new Pair<>(tc, distances));
					tokens.add(tc);
				}
			}
		}
		if(candidate.size() > 0) {
			candidate.sort(AscendingPairListComparator);
			return candidate.get(0).getKey();
		}
		else
			return null;
	}

	public boolean isDelimAndSuffixesSame(String delim, int[] cols, String conflict){
		 HashSet<String>[] ends = properties.endWithValueStrings();
		 boolean flag = true;
		 for(int c = 0; c<cols.length && flag; c++){
			 if(ends[cols[c]].size() == 0)
				 continue;
			 if(ends[cols[c]].size() != 1 || !ends[cols[c]].iterator().next().equals(delim))
				 flag = false;
		 }
		 if(!flag){
			 for(int r=0; r<nrows; r++){
				 ArrayList<Integer> c = new ArrayList<>();
				  for(int ci:cols){
					 if(mapCol[r][ci] !=-1)
						 c.add(ci);
				 }
				  if(c.size() <=1)
					  continue;
				  int c1 = c.get(0);
				  int rowIndex = mapRow[r][c1];
				 String str;
				 if(conflict == null)
					 str = sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][c1]);
				 else {
					 int conflictIndex = sampleRawIndexes[rowIndex].getRaw().indexOf(conflict, mapCol[r][c1]);
					 if(conflictIndex!=-1)
					 	str =sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][c1], conflictIndex);
					 else
						 str = sampleRawIndexes[rowIndex].getRaw().substring(mapCol[r][c1]);
				 }
				 flag = true;
				 if(str.length() > 0) {
					 String[] strValues = str.split(delim, -1);
					 for(int ci=0; ci<c.size() && flag; ci++){
						 if(mapCol[r][c.get(ci)]!=-1)
							 flag = mappingValues.compareCellValue(r, c.get(ci), strValues[ci]);
					 }
				 }
				 if(!flag)
					 break;
			 }
		 }
		 return flag;
	}
	private ArrayList<String> stringTokenize(String str, int tokenLength) {
		ArrayList<String> result = new ArrayList<>();
		HashSet<String> tokenSet = new HashSet<>();
		for(int i = 0; i <= str.length() - tokenLength; i++) {
			String token = str.substring(i, i + tokenLength);
			if(!token.contains(Lop.OPERAND_DELIMITOR) && !tokenSet.contains(token)) {
				result.add(token);
				tokenSet.add(token);
			}
		}
		return result;
	}

	private ArrayList<String> optimalKeyPattern(ArrayList<String> keys, ArrayList<String> prefixes) {
		ArrayList<ArrayList<String>> keysList = new ArrayList<>();
		for(int i = 0; i < keys.size() - 1; i++) {
			String[] keyList = keys.get(i).split("\\s+");
			ArrayList<String> orderedKeys = new ArrayList<>();
			for(int j = 0; j < keyList.length; j++)
				orderedKeys.add(keyList[j]);
			keysList.add(orderedKeys);
		}
		int lastIndex = keys.size() - 1;
		String[] keyList = keys.get(lastIndex).split("\\s+");
		if(keyList.length == 0){
			return keys;
		}
		StringBuilder sbToken = new StringBuilder(keyList[keyList.length - 1]);
		StringBuilder sbSource = new StringBuilder(keys.get(lastIndex));
		int index = sbSource.reverse().indexOf(sbToken.reverse().toString());
		keyList = keys.get(lastIndex).substring(0, keys.get(lastIndex).length() - index - sbToken.length()).split("\\s+");
		ArrayList<String> orderedKeys = new ArrayList<>();
		for(int j = 0; j < keyList.length; j++)
			orderedKeys.add(keyList[j]);
		if(orderedKeys.size() > 0) {
			keysList.add(orderedKeys);
			orderedKeys = new ArrayList<>();
		}
		orderedKeys.add(sbToken.reverse().toString());
		keysList.add(orderedKeys);

		ArrayList<ArrayList<String>>[] fullList = new ArrayList[keysList.size()];
		for(int i = 0; i < keysList.size() - 1; i++)
			fullList[i] = selfPropagate(keysList.get(i));

		ArrayList<ArrayList<String>> tmpLastKey = new ArrayList<>();
		tmpLastKey.add(keysList.get(keysList.size() - 1));
		fullList[keysList.size() - 1] = tmpLastKey;

		ArrayList<ArrayList<String>> candidates = fullList[0];

		for(int i = 1; i < keysList.size(); i++) {
			if(candidates.size() * fullList[i].size() > 500000) {
				ArrayList<ArrayList<String>> tmpCandidates = new ArrayList<>();
				for(ArrayList<String> tmpList : candidates) {
					ArrayList<String> tmpRemainList = new ArrayList<>();
					for(String s : tmpList)
						tmpRemainList.add(s);
					for(int j = i; j < keys.size(); j++)
						tmpRemainList.add(keys.get(j));

					tmpCandidates.add(tmpRemainList);
				}
				candidates = new ArrayList<>();
				ArrayList<String> tmp = new ArrayList<>();
				ArrayList<String> update = checkPattern(tmpCandidates, prefixes).getKey();
				for(int j = 0; j < update.size() - (keys.size() - i); j++) {
					tmp.add(update.get(j));
				}
				candidates.add(tmp);
				candidates = cartesianProduct(candidates, fullList[i]);
			}
			else
				candidates = cartesianProduct(candidates, fullList[i]);
		}
		Pair<ArrayList<String>, Boolean> update = checkPattern(candidates, prefixes);
		if(update.getValue())
			return update.getKey();
		else
			return keys;
	}
	private Pair<ArrayList<String>, Boolean> checkPattern(ArrayList<ArrayList<String>> candidates, ArrayList<String> prefixes) {
		candidates.sort(AscendingArrayOfStringComparator);
		int index = -1;
		for(int i = 0; i < candidates.size(); i++){
			boolean tmpCheck = true;
			for(int j = 0; j< prefixes.size() && tmpCheck; j++){
				tmpCheck = getIndexOfKeyPatternOnString(prefixes.get(j), candidates.get(i),0) == prefixes.get(j).length();
			}
			if(tmpCheck){
				index = i;
				break;
			}
		}
		if(index!=-1)
			return new Pair<>(candidates.get(index), true);
		else
			return new Pair<>(new ArrayList<>(), false);
	}
	private ArrayList<ArrayList<String>> cartesianProduct(ArrayList<ArrayList<String>> list1, ArrayList<ArrayList<String>> list2) {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		for(ArrayList<String> stringArrayList : list1) {
			for(ArrayList<String> strings : list2) {
				ArrayList<String> tmpList = new ArrayList<>();
				for(String s : stringArrayList)
					if(s.length() > 0)
						tmpList.add(s);
				for(String s : strings)
					if(s.length() > 0)
						tmpList.add(s);
				result.add(tmpList);
			}
		}
		return result;
	}
	private ArrayList<ArrayList<String>> selfPropagate(ArrayList<String> list) {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		int n = list.size();
		int allMasks = (1 << n);
		for(int i = 1; i < allMasks; i++) {
			ArrayList<String> tmp = new ArrayList<>();
			for(int j = 0; j < n; j++) {
				if((i & (1 << j)) > 0)
					tmp.add(list.get(j));

			}
			result.add(tmp);
		}
		ArrayList<String> tmp = new ArrayList<>();
		tmp.add("");
		result.add(tmp);
		result.sort(AscendingArrayOfStringComparator);
		return result;
	}

	Comparator<ArrayList<String>> AscendingArrayOfStringComparator = new Comparator<ArrayList<String>>() {
		@Override
		public int compare(ArrayList<String> strings, ArrayList<String> t1) {
			return Integer.compare(strings.size(), t1.size());
		}
	};
	Comparator<String> AscendingStringLengthComparator = new Comparator<String>() {
		@Override
		public int compare(String s, String t1) {
			return s.length() - t1.length();
		}
	};
	Comparator<Pair<String, Integer>> AscendingPairStringComparator = new Comparator<Pair<String, Integer>>() {
		@Override
		public int compare(Pair<String, Integer> stringIntegerPair, Pair<String, Integer> t1) {
			return stringIntegerPair.getKey().length() - t1.getKey().length();
		}
	};

	Comparator<Pair<String, ArrayList<Integer>>> AscendingPairListComparator = new Comparator<Pair<String, ArrayList<Integer>>>() {
		@Override
		public int compare(Pair<String, ArrayList<Integer>> stringArrayListPair, Pair<String, ArrayList<Integer>> t1) {
			boolean flag = true;
			for(int i=0; i< stringArrayListPair.getValue().size() && flag; i++){
				flag = stringArrayListPair.getValue().get(i) > t1.getValue().get(i);
			}
			if(flag)
				return 1;
			else
				return -1;
		}
	};
}
