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

		// ref to Table 1:
		if(mappingProperties.getRecordProperties() == MappingProperties.RecordProperties.SINGLELINE) {

			// #1
			if(rowIndexStructure.getProperties() == RowIndexStructure.IndexProperties.Identity && colIndexStructure.getProperties() == ColIndexStructure.IndexProperties.Identity) {
				KeyTrie[] colKeyPattern;
				// TODO: change method name from buildColsKeyPatternSingleRow to buildColPatternRowIdentity
				colKeyPattern = buildColsKeyPatternSingleRow();
				properties.setColKeyPattern(colKeyPattern);
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
				int begin = Integer.parseInt(colIndexStructure.getColIndexBegin());
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

				// build key pattern for row index
				int numberOfSelectedCols = 3;
				int begin = Integer.parseInt(rowIndexStructure.getRowIndexBegin());
				boolean check, flagReconstruct;
				int[] selectedRowIndex = new int[2];
				KeyTrie rowKeyPattern = null;

				// Select two none zero row as a row index candidate
				int index = 0;
				for(int r = 1; r < nrows; r++) {
					for(int c = 0; c < ncols; c++)
						if(mapRow[r][c] != -1) {
							selectedRowIndex[index++] = r;
							break;
						}
					if(index > 1)
						break;
				}

				for(int c = 0; c < Math.min(numberOfSelectedCols, ncols); c++) {
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
					if(c == numberOfSelectedCols - 1) {
						ArrayList<String> rowPrefixStrings = new ArrayList<>();
						MappingTrie rowTrie = new MappingTrie();
						rowKeyPattern = new KeyTrie();
						for(int si : selectedRowIndex) {
							for(int ci = 0; ci < ncols; ci++) {
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
				begin = Integer.parseInt(colIndexStructure.getColIndexBegin());
				int[] selectedColIndex = new int[2];
				KeyTrie colKeyPattern = null;

				// Select two none zero row as a row index candidate
				index = 0;
				for(int c = 0; c < ncols; c++) {
					for(int r = 1; r < nrows; r++)
						if(mapRow[r][c] != -1) {
							selectedColIndex[index++] = c;
							break;
						}
					if(index > 1)
						break;
				}

				for(int c = 0; c < Math.min(numberOfSelectedCols, ncols); c++) {
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
					if(c == numberOfSelectedCols - 1) {
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

		// check row-index Identity, the identity properties available just for
		// exist and partially exist mapped values
		if(mappingProperties.getDataProperties() != MappingProperties.DataProperties.NOTEXIST) {
			boolean identity = false;
			int missedCount = 0;
			for(int r = 0; r < nrows; r++)
				missedCount += ncols - mostCommonScore(mapRow[r]);

			if(mappingProperties.getRepresentationProperties() == MappingProperties.RepresentationProperties.SYMMETRIC || mappingProperties.getRepresentationProperties() == MappingProperties.RepresentationProperties.SKEWSYMMETRIC)
				missedCount -= (nrows - 1) * (ncols - 1);

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
			int beginIndex = bitSet.nextSetBit(0);
			for(int i = bitSet.nextSetBit(beginIndex + 1); i != -1 && isSeqScatter; i = bitSet.nextSetBit(i + 1))
				isSeqScatter = i == ++beginIndex;
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
			else if(isSeqScatter) {
				rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.SeqScatter);
				return rowIndexStructure;
			}
		}

		//		if(isCellWise && isExist) {
		//			rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.CELLWISEEXIST);
		//		}
		//		else if(!isCellWise && isExist) {
		//			rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.ROWWISEEXIST);
		//		}
		//		else if(isCellWise && !isExist) {
		//
		//		}

		//		RowIndexStructure rowIndexStructure = null;
		//		int numberOfSelectedCols = 3;
		//		MatrixBlock rowIndexMB = new MatrixBlock(nrows, ncols, false);
		//		int scol = Math.min(ncols - numberOfSelectedCols, ncols);
		//		for(int r=0; r<nrows; r++)
		//			for(int c=scol; c<ncols;c++){
		//				if(mapRow)
		//			}
		//
		//
		//
		//
		//
		//
		//
		//
		//
		//		///////////////////////////////////////////////////////////////////////
		//		boolean flag;
		//		int numberOfSelectedCols = 3;
		//		int begin = 0;
		//		boolean check, flagReconstruct;
		//		int[] selectedRowIndex = new int[2];
		//		HashSet<Integer> beginPos = new HashSet<>();
		//		KeyTrie rowPattern = null;
		//
		//		// Select two none zero row as a row index candidate
		//		int index = 0;
		//		for(int r = 1; r < nrows; r++) {
		//			for(int c = 0; c < ncols; c++)
		//				if(mapRow[r][c] != -1) {
		//					selectedRowIndex[index++] = r;
		//					break;
		//				}
		//			if(index > 1)
		//				break;
		//		}

		//		// CELLWISEEXIST: when row index exist in each cell value
		//		for(int c = 0; c < Math.min(numberOfSelectedCols, ncols); c++) {
		//
		//			Pair<ArrayList<String>, ArrayList<Integer>> colPrefixString = extractAllPrefixStringsOfAColSingleLine(c, false);
		//			ArrayList<String> prefixStrings = colPrefixString.getKey();
		//			ArrayList<Integer> prefixStringRowIndexes = colPrefixString.getValue();
		//			ArrayList<RawIndex> prefixRawIndex = new ArrayList<>();
		//
		//			MappingTrie trie = new MappingTrie();
		//			int ri = 0;
		//			for(String ps : prefixStrings)
		//				trie.reverseInsert(ps, prefixStringRowIndexes.get(ri++));
		//
		//			do {
		//				flag = trie.reConstruct();
		//			}
		//			while(flag);
		//
		//			ArrayList<ArrayList<String>> keyPatterns = trie.getAllSequentialKeys();
		//			for(ArrayList<String> kp : keyPatterns) {
		//				for(String ps : prefixStrings) {
		//					StringBuilder sb = new StringBuilder();
		//					int currPos = 0;
		//					for(String k : kp) {
		//						sb.append(ps.substring(currPos, ps.indexOf(k, currPos)));
		//						currPos += sb.length() + k.length();
		//					}
		//					prefixRawIndex.add(new RawIndex(sb.toString()));
		//				}
		//			}
		//
		//			flag = checkPrefixRowIndex(c, begin, prefixRawIndex);
		//			if(!flag) {
		//				begin = 1;
		//				flag = checkPrefixRowIndex(c, begin, prefixRawIndex);
		//			}
		//			if(!flag) {
		//				beginPos.clear();
		//				break;
		//			}
		//			else
		//				beginPos.add(begin);
		//			if(c == numberOfSelectedCols - 1) {
		//				ArrayList<String> rowPrefixStrings = new ArrayList<>();
		//				MappingTrie rowTrie = new MappingTrie();
		//				rowPattern = new KeyTrie();
		//				for(int si : selectedRowIndex) {
		//					for(int ci = 0; ci < ncols; ci++) {
		//						int cri = mapRow[si][ci];
		//						if(cri != -1) {
		//							String str = sampleRawIndexes.get(cri).getSubString(0, mapCol[si][ci]);
		//							RawIndex rawIndex = new RawIndex(str);
		//							Pair<Integer, Integer> pair = rawIndex.findValue(si + begin);
		//							if(pair != null) {
		//								String pstr = str.substring(0, pair.getKey());
		//								if(pstr.length() > 0) {
		//									rowPrefixStrings.add(pstr);
		//									rowTrie.insert(pstr, 1);
		//								}
		//								rowPattern.insertSuffixKeys(str.substring(pair.getKey() + pair.getValue()).toCharArray());
		//							}
		//						}
		//					}
		//				}
		//
		//				do {
		//					ArrayList<ArrayList<String>> selectedKeyPatterns = new ArrayList<>();
		//					keyPatterns = rowTrie.getAllSequentialKeys();
		//					check = false;
		//					for(ArrayList<String> keyPattern : keyPatterns) {
		//						boolean newCheck = checkKeyPatternIsUnique(rowPrefixStrings, keyPattern);
		//						check |= newCheck;
		//						if(newCheck)
		//							selectedKeyPatterns.add(keyPattern);
		//					}
		//					if(check)
		//						keyPatterns = selectedKeyPatterns;
		//					else {
		//						flagReconstruct = rowTrie.reConstruct();
		//						if(!flagReconstruct)
		//							break;
		//					}
		//				}
		//				while(!check);
		//
		//				if(keyPatterns.size() == 0) {
		//					ArrayList<ArrayList<String>> kpl = new ArrayList<>();
		//					ArrayList<String> kpli = new ArrayList<>();
		//					kpli.add("");
		//					kpl.add(kpli);
		//					keyPatterns = kpl;
		//				}
		//				rowPattern.setPrefixKeyPattern(keyPatterns);
		//			}
		//		}
		//		if(beginPos.size() == 1) {
		//			rowIndexStructure = new RowIndexStructure();
		//			rowIndexStructure.setProperties(RowIndexStructure.IndexProperties.CELLWISEEXIST);
		//			rowIndexStructure.setKeyPattern(rowPattern);
		//			Integer bpos = beginPos.iterator().next();
		//			if(bpos > 0)
		//				rowIndexStructure.setRowIndexBegin("-" + bpos);
		//			else
		//				rowIndexStructure.setRowIndexBegin("");
		//		}
		//		return rowIndexStructure;
		return rowIndexStructure;
	}

	private ColIndexStructure getColIndexStructure() {
		ColIndexStructure colIndexStructure = new ColIndexStructure();
		int begin = 0;
		boolean colIndexExist = true;
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
						if(begin == -1) {
							int a = 100;
						}
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

		if(nne > nrows * 0.3) {
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

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	private boolean checkPrefixRowIndex(int colIndex, int beginPos, ArrayList<RawIndex> prefixRawIndex) {
		for(int r = 0; r < nrows; r++) {
			int rowIndex = this.mapRow[r][colIndex];
			if(rowIndex != -1) {
				boolean flag = false;
				for(RawIndex ri : prefixRawIndex) {
					if(ri.findValue(r + beginPos) != null) {
						flag = true;
						break;
					}
				}
				if(!flag)
					return false;
			}
		}
		return true;
	}

	public CustomProperties getFormatProperties() {
		return properties;
	}

	private Integer mostCommonScore(int[] list) {
		Map<Integer, Integer> map = new HashMap<>();
		int nan = 0;
		for(Integer t : list) {
			if(t != -1) {
				Integer val = map.get(t);
				map.put(t, val == null ? 1 : val + 1);
			}
			else
				nan++;
		}
		if(map.size() == 0)
			return nan;

		Map.Entry<Integer, Integer> max = null;
		for(Map.Entry<Integer, Integer> e : map.entrySet()) {
			if(max == null || e.getValue() > max.getValue())
				max = e;
		}
		return max.getValue() + nan;
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
	// This implementation is for nested datasets are scattered on multiple lines
	// The following steps are required:
	// 1.  Extract all prefix strings per column
	// 2. Build key pattern tree for each column
	// 3. Build key pattern for end of values

	private ArrayList<ArrayList<String>> findRowDelimiters() {
		ArrayList<ArrayList<String>> keyPattern = new ArrayList<>();
		Hirschberg hirschberg = new Hirschberg();
		int misMatchPenalty = 3;
		int gapPenalty = 2;

		//extract all lines are in record boundary
		ArrayList<String> recordBoundaries = new ArrayList<>();
		BitSet[] tmpUsedLines = new BitSet[nlines];
		BitSet[] usedLines = new BitSet[nlines];
		int[] minList = new int[nrows];
		HashMap<Integer, Integer> maxColPos = new HashMap<>();
		int[] minColPos = new int[nrows];
		for(int r = 0; r < nrows; r++)
			tmpUsedLines[r] = new BitSet();

		for(int r = 0; r < nrows; r++) {
			int min = nlines;
			int minPos = 0;
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] != -1) {
					tmpUsedLines[r].set(mapRow[r][c]);
					if(mapRow[r][c] <= min) {
						min = mapRow[r][c];
						if(minPos != 0)
							minPos = Math.min(minPos, mapCol[r][c]);
						else
							minPos = mapCol[r][c];

					}
					if(maxColPos.containsKey(mapRow[r][c]))
						maxColPos.put(mapRow[r][c], Math.max(maxColPos.get(mapRow[r][c]), mapCol[r][c] + mapLen[r][c]));
					else
						maxColPos.put(mapRow[r][c], mapCol[r][c] + mapLen[r][c]);
				}
			minList[r] = min;
			minColPos[r] = minPos;
		}

		for(int r = 0; r < nrows; r++) {
			usedLines[r] = new BitSet(nlines);
			for(int i = 0; i < nrows; i++) {
				if(i != r)
					usedLines[r].or(tmpUsedLines[i]);
			}
		}

		for(int r = 0; r < nrows; r++) {
			int beginLine = minList[r];
			for(; beginLine >= 0; beginLine--)
				if(usedLines[r].get(beginLine))
					break;

			StringBuilder sb = new StringBuilder();
			beginLine = Math.max(beginLine, 0);

			if(beginLine + 1 == nlines)
				continue;

			Integer subStrPos = 0;
			if(maxColPos.containsKey(beginLine))
				subStrPos = maxColPos.get(beginLine);

			String str = sampleRawIndexes.get(beginLine).getRaw().substring(subStrPos);
			if(str.length() > 0) {
				sb.append(str).append("\n");
			}
			for(int i = beginLine + 1; i < minList[r]; i++) {
				str = sampleRawIndexes.get(i).getRaw();
				if(str.length() > 0)
					sb.append(str).append("\n");
			}

			str = sampleRawIndexes.get(minList[r]).getRaw().substring(0, minColPos[r]);
			if(str.length() > 0)
				sb.append(str);
			recordBoundaries.add(sb.toString());
		}
		recordBoundaries.remove(recordBoundaries.size() - 1);

		String str1 = recordBoundaries.get(0);
		String str2 = recordBoundaries.get(1);
		Pair<ArrayList<String>, String> pattern = hirschberg.getLCS(str1, str2, misMatchPenalty, gapPenalty);
		if(pattern != null) {
			String intersect = pattern.getValue();
			ArrayList<String> intersectPattern = pattern.getKey();
			for(int i = 2; i < recordBoundaries.size(); i++) {
				pattern = hirschberg.getLCS(intersect, recordBoundaries.get(i), misMatchPenalty, gapPenalty);
				if(pattern != null) {
					intersect = pattern.getValue();
					intersectPattern = pattern.getKey();
				}
				else
					intersect = null;
			}
			if(intersect != null && intersect.length() > 0) {
				keyPattern.add(intersectPattern);
				return keyPattern;
			}
		}
		return null;
	}

	// Build key pattern tree for each column
	private KeyTrie[] buildColsKeyPatternMultiRow() {
		Pair<ArrayList<String>[], Pair<Integer, Integer>[]> prefixStrings = extractAllPrefixStringsOfColsMultiLine(true);
		ArrayList<String>[] suffixStrings = extractAllSuffixStringsOfColsMultiLine();

		KeyTrie[] colKeyPattens = new KeyTrie[ncols];
		for(int c = 0; c < ncols; c++) {
			// 1. Build Prefix Key Pattern
			String colDelim = findStartWithIntersectOfStrings(prefixStrings.getKey()[c], prefixStrings.getValue()[c].getKey());

			HashSet<String> intersect = new HashSet<>();
			intersect.add(colDelim);

			KeyTrie trie = new KeyTrie(colDelim);
			ArrayList<Pair<ArrayList<String>, ArrayList<String>>> remainedPrefixes = new ArrayList<>();
			boolean check;
			do {
				ArrayList<ArrayList<String>> keyPatterns = trie.getPrefixKeyPatterns();
				check = false;
				for(ArrayList<String> keyPattern : keyPatterns) {
					boolean newCheck = checkKeyPatternIsUnique(prefixStrings.getKey()[c], keyPattern);
					check |= newCheck;
					if(newCheck) {
						trie.setAPrefixPath(keyPattern);
					}
				}

				if(!check) {
					remainedPrefixes.clear();
					boolean flag = true;
					for(ArrayList<String> keyPattern : keyPatterns) {
						ArrayList<String> remainedPrefix = new ArrayList<>();
						for(String ps : prefixStrings.getKey()[c])
							remainedPrefix.add(getRemainedSubstring(ps, keyPattern));

						intersect = findStartWithIntersectOfStrings(remainedPrefix);
						if(intersect != null) {
							trie.insertPrefixKeysConcurrent(intersect);
						}
						else {
							remainedPrefixes.add(new Pair<>(keyPattern, remainedPrefix));
							flag = false;
							break;
						}
					}
					if(!flag)
						break;
				}
			}
			while(!check);

			// Suffix pattern is based on char, so we need to extract all chars of a string
			for(String suffix : suffixStrings[c]) {
				trie.insertSuffixKeys(suffix.toCharArray());
			}
			colKeyPattens[c] = trie;
		}
		return colKeyPattens;
	}

	// Extract prefix strings:
	private Pair<ArrayList<String>[], Pair<Integer, Integer>[]> extractAllPrefixStringsOfColsMultiLine(boolean reverse) {

		ArrayList<String>[] result = new ArrayList[ncols];
		Pair<Integer, Integer>[] minmax = new Pair[ncols];
		BitSet[] tmpUsedLines = new BitSet[nlines];
		BitSet[] usedLines = new BitSet[nlines];
		for(int r = 0; r < nrows; r++)
			tmpUsedLines[r] = new BitSet();

		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] != -1)
					tmpUsedLines[r].set(mapRow[r][c]);

		for(int r = 0; r < nrows; r++) {
			usedLines[r] = new BitSet(nlines);
			for(int i = 0; i < nrows; i++) {
				if(i != r)
					usedLines[r].or(tmpUsedLines[i]);
			}
		}

		// extract prefix strings
		for(int c = 0; c < ncols; c++) {
			result[c] = new ArrayList<>();
			int min = 0;
			int max = 0;
			for(int r = 0; r < nrows; r++) {
				int rowIndex = mapRow[r][c];
				if(rowIndex == -1)
					continue;
				StringBuilder sb = new StringBuilder();
				int lastLine = 0;

				for(int i = rowIndex - 1; i >= 0; i--)
					if(usedLines[r].get(i)) {
						lastLine = i;
						break;
					}
				for(int i = lastLine; i < rowIndex; i++) {
					if(sampleRawIndexes.get(i).getRawLength() > 0)
						sb.append(sampleRawIndexes.get(i).getRaw()).append("\n");
				}
				String str = sampleRawIndexes.get(rowIndex).getSubString(0, mapCol[r][c]);
				if(str.length() > 0 && !str.equals("\n"))
					sb.append(str);
				else if(lastLine < rowIndex)
					sb.deleteCharAt(sb.length() - 1);

				if(reverse)
					result[c].add(sb.reverse().toString());
				else
					result[c].add(sb.toString());
				max = Math.max(max, sb.length());
				if(sb.length() < min || min == 0)
					min = sb.length();
				minmax[c] = new Pair<>(min, max);
			}
		}
		return new Pair<>(result, minmax);
	}

	private String findStartWithIntersectOfStrings(ArrayList<String> strList, int minLength) {
		StringBuilder sb = new StringBuilder();
		int i = 0;
		boolean flag = true;
		do {
			char ch = strList.get(0).charAt(i);
			for(int j = 1; j < Math.min(strList.size(), minLength); j++) {
				char cch = strList.get(j).charAt(i);
				if(ch != cch || ch == '\n') {
					flag = false;
					break;
				}
			}
			if(flag)
				sb.append(ch);
			i++;
		}
		while(flag && i < minLength);
		return sb.toString();

	}

	private HashSet<String> findStartWithIntersectOfStrings(ArrayList<String> strList) {
		// 1. Extract all substrings
		// 2. Find intersection of substrings

		HashSet<String>[] substrings = new HashSet[strList.size()];
		for(int i = 0; i < strList.size(); i++)
			substrings[i] = new HashSet<>();

		for(int w = windowSize; w > 2; w--) {
			for(int i = 0; i < strList.size(); i++) {
				substrings[i].clear();
				substrings[i].addAll(getAllSubstringsOfAString(strList.get(i), w));
			}

			HashSet<String> totalIntersect = new HashSet<>(substrings[0]);
			for(int r = 1; r < substrings.length; r++)
				totalIntersect.retainAll(substrings[r]);

			if(totalIntersect.size() > 0)
				return totalIntersect;

		}
		return null;
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

	private ArrayList<String> getAllSubstringsOfAString(String str, int size) {
		ArrayList<String> result = new ArrayList<>();
		if(str == null)
			return result;
		for(int i = 0; i <= str.length() - size; i++) {
			String s = str.substring(i, i + size);
			if(!s.contains("\n"))
				result.add(s);
		}
		return result;
	}

	private String getRemainedSubstring(String str, ArrayList<String> keys) {
		boolean flag = true;
		int currPos = 0;
		for(String k : keys) {
			int index = str.indexOf(k, currPos);
			if(index != -1)
				currPos = index + k.length();
			else {
				flag = false;
				break;
			}
		}
		if(flag)
			return str.substring(currPos);
		else
			return null;
	}

	private ArrayList<String>[] extractAllSuffixStringsOfColsMultiLine() {
		ArrayList<String>[] result = new ArrayList[ncols];
		for(int c = 0; c < ncols; c++) {
			result[c] = new ArrayList<>();

			for(int r = 0; r < nrows; r++) {
				int rowIndex = mapRow[r][c];
				if(rowIndex == -1)
					continue;
				StringBuilder sb = new StringBuilder();
				String str = sampleRawIndexes.get(rowIndex).getRaw().substring(mapCol[r][c] + mapLen[r][c]);
				boolean enter = false;
				if(str.length() > 0) {
					sb.append(str);
					enter = true;
				}

				for(int i = rowIndex + 1; i < nlines; i++) {
					str = sampleRawIndexes.get(i).getRaw().substring(0, Math.min(sampleRawIndexes.get(i).getRawLength(), suffixStringLength));
					if(str.length() > 0 && !enter) {
						sb.append(str);
						break;
					}
				}
				if(sb.length() > 0)
					sb.deleteCharAt(sb.length() - 1);
				result[c].add(sb.toString());
			}
		}
		return result;
	}

}
