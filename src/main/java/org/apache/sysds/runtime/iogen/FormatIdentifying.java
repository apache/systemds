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

import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class FormatIdentifying {

	private int[][] mapRow;
	private int[][] mapCol;
	private int[][] mapLen;
	private int actualValueCount;
	private MappingProperties mappingProperties;
	private ArrayList<RawIndex> sampleRawIndexes;

	private static int nrows;
	private static int ncols;
	private int nlines;

	private int windowSize = 20;
	private int suffixStringLength = 50;
	private ReaderMapping mappingValues;
	private CustomProperties properties;

	public FormatIdentifying(String raw, MatrixBlock matrix) throws Exception {
		this.mappingValues = new ReaderMapping(raw, matrix);
		this.runIdentification();
	}

	public FormatIdentifying(String raw, FrameBlock frame) throws Exception {
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
			if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.Identity && colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.Identity) {
				KeyTrie[] colKeyPatterns;
				colKeyPatterns = buildColsKeyPatternSingleRow();
				properties.setColKeyPatterns(colKeyPatterns);
			}

			// #2
			else if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.Identity && colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {
				// find cell-index and value separators
				RawIndex raw = null;
				for(int c = 0; c < ncols; c++) {
					if(mapCol[0][c] != -1) {
						raw = sampleRawIndexes.get(mapRow[0][c]);
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
			if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.CellWiseExist && colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {

				if(mappingProperties.getDataProperties() != MappingProperties.DataProperties.NOTEXIST) {
					KeyTrie valueKeyPattern = buildValueKeyPattern();
					properties.setValueKeyPattern(valueKeyPattern);
				}

				// build key pattern for row index
				int numberOfSelectedCols = (int) (ncols * 0.1);
				int numberOfSelectedRows = (int) (nrows * 0.1);
				numberOfSelectedRows = numberOfSelectedRows == 0 ? nrows - 1 : numberOfSelectedRows;
				numberOfSelectedCols = numberOfSelectedCols == 0 ? ncols - 1 : numberOfSelectedCols;
				int begin = rowIndexStructure.getRowIndexBegin();
				boolean check, flagReconstruct;
				int[] selectedRowIndex = new int[numberOfSelectedRows];
				KeyTrie rowKeyPattern = null;

				// Select two none zero row as a row index candidate
				int index = 0;
				for(int r = 1; r < nrows; r++) {
					for(int c = 0; c < ncols; c++)
						if(mapRow[r][c] != -1) {
							selectedRowIndex[index++] = r;
							break;
						}
					if(index >= numberOfSelectedRows)
						break;
				}

				for(int c = ncols - 1; c >= Math.max(ncols - numberOfSelectedCols, 0); c--) {
					Pair<ArrayList<String>, ArrayList<Integer>> colPrefixString = extractAllPrefixStringsOfAColSingleLine(c, false);
					ArrayList<String> prefixStrings = colPrefixString.getKey();
					ArrayList<Integer> prefixStringRowIndexes = colPrefixString.getValue();
					ArrayList<RawIndex> prefixRawIndex = new ArrayList<>();

					MappingTrie trie = new MappingTrie();
					int ri = 0;
					for(String ps : prefixStrings)
						trie.reverseInsert(ps, prefixStringRowIndexes.get(ri++));

					do {
						flagReconstruct = trie.reConstruct();
					}
					while(flagReconstruct);

					ArrayList<ArrayList<String>> keyPatterns = trie.getAllSequentialKeys();
					for(ArrayList<String> kp : keyPatterns) {
						for(String ps : prefixStrings) {
							StringBuilder sb = new StringBuilder();
							int currPos = 0;
							for(String k : kp) {
								sb.append(ps.substring(currPos, ps.indexOf(k, currPos)));
								currPos += sb.length() + k.length();
							}
							prefixRawIndex.add(new RawIndex(sb.toString()));
						}
					}
					if(c == ncols - 1) {
						ArrayList<String> rowPrefixStrings = new ArrayList<>();
						MappingTrie rowTrie = new MappingTrie();
						rowKeyPattern = new KeyTrie();
						for(int si : selectedRowIndex) {
							for(int ci = ncols - 1; ci >= 0; ci--) {
								int cri = mapRow[si][ci];
								if(cri != -1) {
									String str = sampleRawIndexes.get(cri).getSubString(0, mapCol[si][ci]);
									RawIndex rawIndex = new RawIndex(str);
									Pair<Integer, Integer> pair = rawIndex.findValue(si + begin);
									if(pair != null) {
										String pstr = str.substring(0, pair.getKey());
										if(pstr.length() > 0) {
											rowPrefixStrings.add(pstr);
											rowTrie.insert(pstr, 1);
										}
										rowKeyPattern.insertSuffixKeys(str.substring(pair.getKey() + pair.getValue()).toCharArray());
									}
								}
							}
						}

						do {
							ArrayList<ArrayList<String>> selectedKeyPatterns = new ArrayList<>();
							keyPatterns = rowTrie.getAllSequentialKeys();
							check = false;
							for(ArrayList<String> keyPattern : keyPatterns) {
								boolean newCheck = checkKeyPatternIsUnique(rowPrefixStrings, keyPattern);
								check |= newCheck;
								if(newCheck)
									selectedKeyPatterns.add(keyPattern);
							}
							if(check)
								keyPatterns = selectedKeyPatterns;
							else {
								flagReconstruct = rowTrie.reConstruct();
								if(!flagReconstruct)
									break;
							}
						}
						while(!check);

						if(keyPatterns.size() == 0) {
							ArrayList<ArrayList<String>> kpl = new ArrayList<>();
							ArrayList<String> kpli = new ArrayList<>();
							kpli.add("");
							kpl.add(kpli);
							keyPatterns = kpl;
						}
						rowKeyPattern.setPrefixKeyPattern(keyPatterns);
					}
				}
				rowIndexStructure.setKeyPattern(rowKeyPattern);

				// build key pattern for column index
				begin = colIndexStructure.getColIndexBegin();
				int[] selectedColIndex = new int[numberOfSelectedCols];
				KeyTrie colKeyPattern = null;

				// Select two none zero row as a row index candidate
				index = 0;
				for(int c = ncols - 1; c >= 0; c--) {
					for(int r = 1; r < nrows; r++)
						if(mapRow[r][c] != -1) {
							selectedColIndex[index++] = c;
							break;
						}
					if(index >= numberOfSelectedCols)
						break;
				}

				for(int c = ncols - 1; c >= Math.max(ncols - numberOfSelectedCols, 0); c--) {
					Pair<ArrayList<String>, ArrayList<Integer>> colPrefixString = extractAllPrefixStringsOfAColSingleLine(c, false);
					ArrayList<String> prefixStrings = colPrefixString.getKey();
					ArrayList<Integer> prefixStringRowIndexes = colPrefixString.getValue();
					ArrayList<RawIndex> prefixRawIndex = new ArrayList<>();

					MappingTrie trie = new MappingTrie();
					int ri = 0;
					for(String ps : prefixStrings)
						trie.reverseInsert(ps, prefixStringRowIndexes.get(ri++));

					do {
						flagReconstruct = trie.reConstruct();
					}
					while(flagReconstruct);

					ArrayList<ArrayList<String>> keyPatterns = trie.getAllSequentialKeys();
					for(ArrayList<String> kp : keyPatterns) {
						for(String ps : prefixStrings) {
							StringBuilder sb = new StringBuilder();
							int currPos = 0;
							for(String k : kp) {
								sb.append(ps.substring(currPos, ps.indexOf(k, currPos)));
								currPos += sb.length() + k.length();
							}
							prefixRawIndex.add(new RawIndex(sb.toString()));
						}
					}
					if(c == ncols - 1) {
						ArrayList<String> colPrefixStrings = new ArrayList<>();
						MappingTrie colTrie = new MappingTrie();
						colKeyPattern = new KeyTrie();
						for(int si : selectedColIndex) {
							for(int ir = 0; ir < nrows; ir++) {
								int cri = mapRow[ir][si];
								if(cri != -1) {
									String str = sampleRawIndexes.get(cri).getSubString(0, mapCol[ir][si]);
									RawIndex rawIndex = new RawIndex(str);
									Pair<Integer, Integer> pair = rawIndex.findValue(si + begin);
									if(pair != null) {
										String pstr = str.substring(0, pair.getKey());
										if(pstr.length() > 0) {
											colPrefixStrings.add(pstr);
											colTrie.insert(pstr, 1);
										}
										colKeyPattern.insertSuffixKeys(str.substring(pair.getKey() + pair.getValue()).toCharArray());
									}
								}
							}
						}

						do {
							ArrayList<ArrayList<String>> selectedKeyPatterns = new ArrayList<>();
							keyPatterns = colTrie.getAllSequentialKeys();
							check = false;
							for(ArrayList<String> keyPattern : keyPatterns) {
								boolean newCheck = checkKeyPatternIsUnique(colPrefixStrings, keyPattern);
								check |= newCheck;
								if(newCheck)
									selectedKeyPatterns.add(keyPattern);
							}
							if(check)
								keyPatterns = selectedKeyPatterns;
							else {
								flagReconstruct = colTrie.reConstruct();
								if(!flagReconstruct)
									break;
							}
						}
						while(!check);

						if(keyPatterns.size() == 0) {
							ArrayList<ArrayList<String>> kpl = new ArrayList<>();
							ArrayList<String> kpli = new ArrayList<>();
							kpli.add("");
							kpl.add(kpli);
							keyPatterns = kpl;
						}
						colKeyPattern.setPrefixKeyPattern(keyPatterns);
					}
				}
				colIndexStructure.setKeyPattern(colKeyPattern);
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
							textTrie.insert(prefix.substring(j, j + Math.min(minSubStringLength, prefix.length() - j)), i);
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
						String str = new StringBuilder(prefixSuffixBeginEndCells.get(i).getValue()).reverse().toString();
						int indexBeginString = str.indexOf(reverseBeginString);
						if(indexBeginString != -1) {
							for(int j = indexBeginString + reverseBeginString.length(); j < str.length(); j++) {
								if(str.charAt(j) == '\n')
									indexBeginString++;
								else
									break;
							}
							minSuffixStringLength = Math.min(minSuffixStringLength, indexBeginString);
							suffixes.add(str.substring(indexBeginString + reverseBeginString.length()));
						}
						else
							suffixes.add(str);
					}
					StringBuilder sbEndString = new StringBuilder();
					for(int i = 0; i < minSuffixStringLength; i++) {
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
						endString = sbEndString.reverse().toString();
					updateMapsAndExtractAllSuffixStringsOfColsMultiLine(beginString, endString);
					rowIndexStructure.setSeqBeginString(beginString);
					rowIndexStructure.setSeqEndString(endString);
					KeyTrie[] colKeyPatterns;
					colKeyPatterns = buildColsKeyPatternSingleRow();
					properties.setColKeyPatterns(colKeyPatterns);
				}
				else {
					// TODO: extend sequential scattered format algorithm for heterogeneous structures
				}
			}
		}

		if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.CellWiseExist || colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {
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
					list.add(sampleRawIndexes.get(i));
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
				RawIndex raw = sampleRawIndexes.get(mapRow[r][colIndex]);
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
		RawIndex raw = sampleRawIndexes.get(rowIndex);
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
		RawIndex raw = sampleRawIndexes.get(rowIndex);
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
			beginPos = sampleRawIndexes.get(beginLine).getRawLength();
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] == beginLine)
					beginPos = Math.min(beginPos, mapCol[r][c]);

				if(mapRow[r][c] == endLine)
					endPos = Math.max(endPos, mapCol[r][c] + mapLen[r][c]);
			}
			StringBuilder sbPrefix = new StringBuilder();
			if(lastLine != beginLine)
				sbPrefix.append(sampleRawIndexes.get(lastLine).getRaw().substring(lastPos)).append("\n");

			for(int i = lastLine + 1; i < beginLine; i++)
				sbPrefix.append(sampleRawIndexes.get(i).getRaw()).append("\n");
			sbPrefix.append(sampleRawIndexes.get(beginLine).getRaw().substring(0, beginPos));

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

	private Integer mostCommonValue(int[] list) {
		Map<Integer, Integer> map = new HashMap<>();
		for(Integer t : list) {
			if(t != -1) {
				Integer val = map.get(t);
				map.put(t, val == null ? 1 : val + 1);
			}
		}
		if(map.size() == 0)
			return -1;

		Map.Entry<Integer, Integer> max = null;
		for(Map.Entry<Integer, Integer> e : map.entrySet()) {
			if(max == null || e.getValue() > max.getValue())
				max = e;
		}
		return max.getKey();
	}

	private KeyTrie[] buildColsKeyPatternSingleRow() {
		Pair<ArrayList<String>[], ArrayList<Integer>[]> prefixStrings = extractAllPrefixStringsOfColsSingleLine(false);
		ArrayList<String>[] suffixStrings = extractAllSuffixStringsOfColsSingleLine();
		KeyTrie[] colKeyPattens = new KeyTrie[ncols];

		// Clean prefix strings
		for(int c = 0; c < ncols; c++) {
			ArrayList<String> list = prefixStrings.getKey()[c];
			String token = null;
			boolean flag = true;
			for(int w = 1; w < windowSize && flag; w++) {
				HashSet<String> wts = new HashSet<>();
				for(String s : list) {
					if(s.length() < w)
						flag = false;
					else {
						String subStr = s.substring(s.length() - w);
						if(!subStr.contains(Lop.OPERAND_DELIMITOR))
							wts.add(subStr);
						else
							flag = false;
					}
				}

				if(flag) {
					if(wts.size() == 1)
						token = wts.iterator().next();
					else {
						for(String t : wts) {
							int count = 0;
							for(String s : list) {
								if(s.endsWith(t))
									count++;
							}
							float percent = (float) count / list.size();
							if(percent >= 1)
								token = t;
						}
					}
				}
				else if(wts.size() == 0)
					token = "";
			}
			if(token == null) {
				int[] listLength = new int[nrows];
				for(int r = 0; r < nrows; r++)
					listLength[r] = mapCol[r][c];
				int commonLength = mostCommonValue(listLength);
				if(commonLength == 0) {
					ArrayList<String> newList = new ArrayList<>();
					for(String s : list) {
						if(s.length() == 0)
							newList.add(s);
					}
					prefixStrings.getKey()[c] = newList;
				}
				else
					throw new RuntimeException("can't build a key pattern for the column: " + c);
			}
			else if(token.length() > 0) {
				ArrayList<String> newList = new ArrayList<>();
				for(String s : list) {
					if(s.endsWith(token))
						newList.add(s);
				}
				prefixStrings.getKey()[c] = newList;
			}
		}

		for(int c = 0; c < ncols; c++) {
			MappingTrie trie = new MappingTrie();
			int ri = 0;
			boolean check;
			boolean flagReconstruct;
			ArrayList<ArrayList<String>> keyPatterns = null;

			for(String ps : prefixStrings.getKey()[c])
				trie.reverseInsert(ps, prefixStrings.getValue()[c].get(ri++));

			if(trie.getRoot().getChildren().size() == 1) {
				String[] splitPattern = prefixStrings.getKey()[c].get(0).split(Lop.OPERAND_DELIMITOR);
				ArrayList<String> reverseSplitPattern = new ArrayList<>();
				for(String ps : splitPattern)
					if(ps.length() > 0)
						reverseSplitPattern.add(ps);
				if(reverseSplitPattern.size() == 0)
					reverseSplitPattern.add("");

				int maxPatternLength = reverseSplitPattern.size();
				check = false;
				for(int sp = 0; sp < maxPatternLength; sp++) {
					ArrayList<String> shortPattern = new ArrayList<>();
					for(int spi = maxPatternLength - sp - 1; spi < maxPatternLength; spi++) {
						shortPattern.add(reverseSplitPattern.get(spi));
					}
					check = checkKeyPatternIsUnique(prefixStrings.getKey()[c], shortPattern);
					if(check) {
						keyPatterns = new ArrayList<>();
						keyPatterns.add(shortPattern);
						break;
					}
				}
			}
			else {
				do {
					ArrayList<ArrayList<String>> selectedKeyPatterns = new ArrayList<>();
					keyPatterns = trie.getAllSequentialKeys();
					check = false;
					for(ArrayList<String> keyPattern : keyPatterns) {
						boolean newCheck = checkKeyPatternIsUnique(prefixStrings.getKey()[c], keyPattern);
						check |= newCheck;
						if(newCheck)
							selectedKeyPatterns.add(keyPattern);
					}
					if(check)
						keyPatterns = selectedKeyPatterns;
					else {
						flagReconstruct = trie.reConstruct();
						if(!flagReconstruct)
							break;
					}
				}
				while(!check);
			}

			if(check) {
				colKeyPattens[c] = new KeyTrie(keyPatterns);
				for(String suffix : suffixStrings[c]) {
					colKeyPattens[c].insertSuffixKeys(suffix.substring(0, Math.min(suffixStringLength, suffix.length())).toCharArray());
				}
			}
		}
		return colKeyPattens;
	}

	private KeyTrie buildValueKeyPattern() {
		int minSelectCols = Math.min(10, ncols);
		ArrayList<String> prefixStrings = new ArrayList<>();
		ArrayList<Integer> rowIndexes = new ArrayList<>();
		ArrayList<String> suffixStrings = new ArrayList<>();

		for(int c = 0; c < minSelectCols; c++) {
			Pair<ArrayList<String>, ArrayList<Integer>> pair = extractAllPrefixStringsOfAColSingleLine(c, false);
			prefixStrings.addAll(pair.getKey());
			rowIndexes.addAll(pair.getValue());
		}

		for(int c = 0; c < minSelectCols; c++) {
			for(int r = 0; r < nrows; r++) {
				int rowIndex = mapRow[r][c];
				if(rowIndex == -1)
					continue;
				String str = sampleRawIndexes.get(rowIndex).getRaw().substring(mapCol[r][c] + mapLen[r][c]);
				suffixStrings.add(str);
			}
		}

		KeyTrie valueKeyPatten = new KeyTrie();
		for(int c = 0; c < ncols; c++) {
			MappingTrie trie = new MappingTrie();
			int ri = 0;
			boolean check;
			boolean flagReconstruct;
			ArrayList<ArrayList<String>> keyPatterns = null;

			int psIndex = 0;
			for(String ps : prefixStrings)
				trie.reverseInsert(ps, rowIndexes.get(psIndex++));

			if(trie.getRoot().getChildren().size() == 1) {
				String[] splitPattern = prefixStrings.get(0).split(Lop.OPERAND_DELIMITOR);
				ArrayList<String> reverseSplitPattern = new ArrayList<>();
				for(String ps : splitPattern)
					if(ps.length() > 0)
						reverseSplitPattern.add(ps);
				if(reverseSplitPattern.size() == 0)
					reverseSplitPattern.add("");

				int maxPatternLength = reverseSplitPattern.size();
				check = false;
				for(int sp = 0; sp < maxPatternLength; sp++) {
					ArrayList<String> shortPattern = new ArrayList<>();
					for(int spi = maxPatternLength - sp - 1; spi < maxPatternLength; spi++) {
						shortPattern.add(reverseSplitPattern.get(spi));
					}
					check = checkKeyPatternIsUnique(prefixStrings, shortPattern);
					if(check) {
						keyPatterns = new ArrayList<>();
						keyPatterns.add(shortPattern);
						break;
					}
				}
			}
			else {
				do {
					ArrayList<ArrayList<String>> selectedKeyPatterns = new ArrayList<>();
					keyPatterns = trie.getAllSequentialKeys();
					check = false;
					for(ArrayList<String> keyPattern : keyPatterns) {
						boolean newCheck = checkKeyPatternIsUnique(prefixStrings, keyPattern);
						check |= newCheck;
						if(newCheck)
							selectedKeyPatterns.add(keyPattern);
					}
					if(check)
						keyPatterns = selectedKeyPatterns;
					else {
						flagReconstruct = trie.reConstruct();
						if(!flagReconstruct)
							break;
					}
				}
				while(!check);
			}

			if(check) {
				valueKeyPatten = new KeyTrie(keyPatterns);
				for(String suffix : suffixStrings) {
					valueKeyPatten.insertSuffixKeys(suffix.substring(0, Math.min(suffixStringLength, suffix.length())).toCharArray());
				}
			}
		}
		return valueKeyPatten;
	}

	// Get all prefix strings of a column
	public Pair<ArrayList<String>[], ArrayList<Integer>[]> extractAllPrefixStringsOfColsSingleLine(boolean reverse) {
		ArrayList<String>[] prefixStrings = new ArrayList[ncols];
		ArrayList<Integer>[] rowIndexes = new ArrayList[ncols];
		for(int c = 0; c < ncols; c++) {
			Pair<ArrayList<String>, ArrayList<Integer>> pair = extractAllPrefixStringsOfAColSingleLine(c, reverse);
			prefixStrings[c] = pair.getKey();
			rowIndexes[c] = pair.getValue();
		}
		return new Pair<>(prefixStrings, rowIndexes);
	}

	public Pair<ArrayList<String>, ArrayList<Integer>> extractAllPrefixStringsOfAColSingleLine(int colIndex, boolean reverse) {
		ArrayList<String> prefixStrings = new ArrayList();
		ArrayList<Integer> rowIndexes = new ArrayList();
		for(int r = 0; r < nrows; r++) {
			int rowIndex = mapRow[r][colIndex];
			if(rowIndex != -1) {
				rowIndexes.add(rowIndex);
				String str = sampleRawIndexes.get(rowIndex).getRemainedTexts(mapCol[r][colIndex]);
				if(reverse)
					prefixStrings.add(new StringBuilder(str).reverse().toString());
				else
					prefixStrings.add(str);
			}
		}
		return new Pair<>(prefixStrings, rowIndexes);
	}

	private ArrayList<String>[] extractAllSuffixStringsOfColsSingleLine() {
		ArrayList<String>[] result = new ArrayList[ncols];
		for(int c = 0; c < ncols; c++) {
			result[c] = new ArrayList<>();
			for(int r = 0; r < nrows; r++) {
				int rowIndex = mapRow[r][c];
				if(rowIndex == -1)
					continue;
				String str = sampleRawIndexes.get(rowIndex).getRaw().substring(mapCol[r][c] + mapLen[r][c]);
				result[c].add(str);
			}
		}
		return result;
	}

	/////////////////////////////////////////////////////////////////////////////
	//                    Methods For Multi Lines Mapping                     //
	////////////////////////////////////////////////////////////////////////////

	private void updateMapsAndExtractAllSuffixStringsOfColsMultiLine(String beginString, String endString) {
		ArrayList<RawIndex> upRawIndexes = new ArrayList<>();
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
			endIndexes.add(new Pair<>(this.sampleRawIndexes.size() - 1, this.sampleRawIndexes.get(this.sampleRawIndexes.size() - 1).getRawLength()));
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
			sb.append(this.sampleRawIndexes.get(beginIndexes.get(i - n).getKey()).getRaw().substring(beginIndexes.get(i - n).getValue()));
			for(int ri = beginIndexes.get(i - n).getKey() + 1; ri < endIndexes.get(j).getKey(); ri++) {
				sb.append(this.sampleRawIndexes.get(ri).getRaw());
			}
			sb.append(this.sampleRawIndexes.get(endIndexes.get(j).getKey()).getRaw().substring(0, endIndexes.get(j).getValue())).append(endToken);
			RawIndex rawIndex = new RawIndex();
			rawIndex.setRaw(sb.toString());
			sb = new StringBuilder();
			j++;
			// update mapping
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] != -1) {
					if(mapRow[r][c] != beginIndexes.get(i - n).getKey())
						this.mapCol[r][c] += this.sampleRawIndexes.get(beginIndexes.get(i - n).getKey()).getRawLength() - beginIndexes.get(i - n)
							.getValue();
					else
						this.mapCol[r][c] -= beginIndexes.get(i - n).getValue();

					for(int ci = beginIndexes.get(i - n).getKey() + 1; ci < this.mapRow[r][c]; ci++)
						this.mapCol[r][c] += this.sampleRawIndexes.get(ci).getRawLength();
					rawIndex.setReservedPositions(mapCol[r][c], mapLen[r][c]);
					this.mapRow[r][c] = r;
				}
			}
			upRawIndexes.add(rawIndex);
			r++;
		}
		this.sampleRawIndexes = upRawIndexes;
	}

	private ArrayList<Pair<Integer, Integer>> getTokenIndexOnMultiLineRecords(String token) {
		ArrayList<Pair<Integer, Integer>> result = new ArrayList<>();

		for(int ri = 0; ri < this.sampleRawIndexes.size(); ri++) {
			String raw = this.sampleRawIndexes.get(ri).getRaw();
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

		for(int ri = 0; ri < this.sampleRawIndexes.size(); ) {
			String raw = this.sampleRawIndexes.get(ri).getRaw();
			int index;
			int fromIndex = 0;
			do {
				index = raw.indexOf(endToken, fromIndex);
				if(index != -1) {
					if(index + endToken.length() + beginToken.length() <= raw.length()) {
						boolean flag = true;
						for(int i = index + endToken.length(), j = 0; i < index + endToken.length() + beginToken.length() && flag; i++, j++) {
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
						if(ri+1 == this.sampleRawIndexes.size())
							break;
						// skip empty rows
						do {
							raw = this.sampleRawIndexes.get(++ri).getRaw();
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

	private boolean checkKeyPatternIsUnique(ArrayList<String> prefixStrings, ArrayList<String> keys) {
		if(keys.size() == 1) {
			String k = keys.get(0);
			if(k.length() == 0)
				return true;
		}

		for(String ps : prefixStrings) {
			int currentPos = 0;
			int patternCount = 0;
			do {
				currentPos = getIndexOfKeyPatternOnString(ps, keys, currentPos).getKey();
				if(currentPos == -1)
					break;
				else {
					patternCount++;
					currentPos++;
				}
			}
			while(true);
			if(patternCount != 1)
				return false;
		}
		return true;
	}

	// Check the sequential list of keys are on a string
	private Pair<Integer, Integer> getIndexOfKeyPatternOnString(String str, ArrayList<String> key, int beginPos) {

		int currPos = beginPos;
		boolean flag = true;
		int startPos = -1;
		for(String k : key) {
			int index = str.indexOf(k, currPos);
			if(index != -1)
				currPos = index + k.length();
			else {
				flag = false;
				break;
			}
			if(startPos == -1)
				startPos = currPos;
		}
		if(flag)
			return new Pair<>(startPos, currPos + key.get(key.size() - 1).length());
		else
			return new Pair<>(-1, -1);
	}
}
