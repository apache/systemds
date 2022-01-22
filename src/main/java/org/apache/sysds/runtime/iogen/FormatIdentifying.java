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
import java.util.BitSet;
import java.util.HashSet;

public class FormatIdentifying {

	private int[][] mapRow;
	private int[]   mapRowPrevious;
	private int[][] mapCol;
	private int[][] mapLen;
	private ArrayList<RawIndex> sampleRawIndexes;

	private static int nrows;
	private static int ncols;
	private int nlines;
	private int windowSize = 20;
	private int suffixStringLength = 100;
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

		mapRow = mappingValues.getMapRow();
		mapCol = mappingValues.getMapCol();
		mapLen = mappingValues.getMapLen();
		sampleRawIndexes = mappingValues.getSampleRawIndexes();
		mapRowPrevious = new int[ncols];

		for(int c=0; c< ncols; c++)
			mapRowPrevious[c] = 0;

		nrows = mappingValues.getNrows();
		ncols = mappingValues.getNcols();
		nlines = mappingValues.getNlines();

		// Check the map row:
		// If all cells of a row mapped to a single line of sample raw, it is a single row mapping
		// If all cells of a row mapped to multiple lines of sample raw, it is a multi row mapping

		boolean isSingleRow = true;
		for(int r=0; r<nrows && isSingleRow; r++){
			int rowIndex = -1;
			int c = 0;
			for(; c<ncols;c++){
				if(mapRow[r][c] !=-1) {
					rowIndex = mapRow[r][c];
					break;
				}
			}
			for(; c< ncols && isSingleRow; c++){
				if(mapRow[r][c] !=-1) {
					isSingleRow = mapRow[r][c] == rowIndex;
				}
			}
		}

		KeyTrie[] colKeyPattern;
		if(isSingleRow){
			colKeyPattern = buildColsKeyPatternSingleRow();
			properties = new CustomProperties(colKeyPattern, CustomProperties.IndexProperties.IDENTIFY);
		}else {

			// Check the row index is a prefix string in sample raw
			// if the row indexes are in the prefix of values, so we need to build a key pattern
			// to extract row indexes
			// to understanding row indexes are in sample raw we check just 3 column of data
			// for build a key pattern ro row indexes we just selected a row
			boolean flag;
			int numberOfSelectedCols = 3;
			int begin = 0;
			boolean check, flagReconstruct;
			int selectedRowIndex = 1;
			HashSet<Integer> beginPos = new HashSet<>();
			KeyTrie rowKeyPattern = null;

			for(int c=0; c< Math.min(numberOfSelectedCols, ncols); c++){
				Pair<ArrayList<String>, ArrayList<Integer>> colPrefixString = extractAllPrefixStringsOfAColSingleLine(c, false);
				ArrayList<String> prefixStrings = colPrefixString.getKey();
				ArrayList<Integer> prefixStringRowIndexes = colPrefixString.getValue();
				ArrayList<RawIndex> prefixRawIndex = new ArrayList<>();

				MappingTrie trie = new MappingTrie();
				int ri = 0;
				for(String ps: prefixStrings )
					trie.reverseInsert(ps, prefixStringRowIndexes.get(ri++));


				do {
					flag = trie.reConstruct();
				}while(flag);

				ArrayList<ArrayList<String>> keyPatterns = trie.getAllSequentialKeys();
				for(ArrayList<String> kp: keyPatterns){
					for(String ps: prefixStrings){
						StringBuilder sb = new StringBuilder();
						int currPos = 0;
						for(String k: kp){
							sb.append(ps.substring(currPos, ps.indexOf(k, currPos)));
							currPos += sb.length() + k.length();
						}
						prefixRawIndex.add(new RawIndex(sb.toString()));
					}
				}

				 flag = checkPrefixRowIndex(c, begin, prefixRawIndex);
				if(!flag) {
					begin = 1;
					flag = checkPrefixRowIndex(c, begin, prefixRawIndex);
				}
				if(!flag) {
					beginPos.clear();
					break;
				}
				else
					beginPos.add(begin);
				if(c== numberOfSelectedCols -1){
					ArrayList<String> rowPrefixStrings = new ArrayList<>();
					MappingTrie rowTrie = new MappingTrie();
					rowKeyPattern = new KeyTrie();
					for(int ci = 0; c < ncols; c++) {
						int cri = mapRow[selectedRowIndex][c];
						if(cri != -1) {
							String str = sampleRawIndexes.get(cri).getSubString(0, mapCol[selectedRowIndex][ci]);
							RawIndex rawIndex = new RawIndex(str);
							Pair<Integer,Integer> pair = rawIndex.findValue(selectedRowIndex+begin);
							if(pair!=null) {
								String pstr = str.substring(0, pair.getKey());
								if(pstr.length() > 0) {
									rowPrefixStrings.add(pstr);
									rowTrie.insert(pstr, 1);
								}
								rowKeyPattern.insertSuffixKeys(str.substring(pair.getKey()+pair.getValue()).toCharArray());
							}
						}
					}

					do {
						keyPatterns = rowTrie.getAllSequentialKeys();
						check = false;
						for(ArrayList<String> keyPattern : keyPatterns) {
							boolean newCheck = checkKeyPatternIsUnique(rowPrefixStrings, keyPattern);
							check |= newCheck;
						}
						if(!check){
							flagReconstruct = trie.reConstruct();
							if(!flagReconstruct)
								break;
						}
					}while(!check);

					rowKeyPattern.setPrefixKeyPattern(keyPatterns);
				}
			}

			if(beginPos.size() == 1){
				colKeyPattern = buildColsKeyPatternSingleRow();
				properties = new CustomProperties(colKeyPattern, CustomProperties.IndexProperties.PREFIX, rowKeyPattern);
			}
			else {
				colKeyPattern = buildColsKeyPatternMultiRow();
				properties = new CustomProperties(colKeyPattern, CustomProperties.IndexProperties.KEY);
			}
		}
	}

	private boolean checkPrefixRowIndex(int colIndex, int beginPos, ArrayList<RawIndex> prefixRawIndex){
		for(int r=0;r<nrows; r++){
			int rowIndex = this.mapRow[r][colIndex];
			if(rowIndex!=-1){
				boolean flag = false;
				for(RawIndex ri: prefixRawIndex) {
					if(ri.findValue(r+ beginPos) != null) {
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

	private KeyTrie[] buildColsKeyPatternSingleRow() {
		Pair<ArrayList<String>[], ArrayList<Integer>[]> prefixStrings = extractAllPrefixStringsOfColsSingleLine(false);
		ArrayList<String>[] suffixStrings = extractAllSuffixStringsOfColsSingleLine();
		KeyTrie[] colKeyPattens = new KeyTrie[ncols];

		for(int c=0; c<ncols; c++) {
			MappingTrie trie = new MappingTrie();
			int ri = 0;
			for(String ps: prefixStrings.getKey()[c])
				trie.reverseInsert(ps, prefixStrings.getValue()[c].get(ri++));

			boolean check;
			boolean flagReconstruct;
			ArrayList<ArrayList<String>> keyPatterns;
			do {
				keyPatterns = trie.getAllSequentialKeys();
				check = false;
				for(ArrayList<String> keyPattern : keyPatterns) {
					boolean newCheck = checkKeyPatternIsUnique(prefixStrings.getKey()[c], keyPattern);
					check |= newCheck;
				}

				if(!check){
					flagReconstruct = trie.reConstruct();
					if(!flagReconstruct)
						break;
				}
			}while(!check);

			if(check){
				colKeyPattens[c] = new KeyTrie(keyPatterns);
				for(String suffix: suffixStrings[c]) {
					colKeyPattens[c].insertSuffixKeys(suffix.substring(0,Math.min(suffixStringLength, suffix.length())).toCharArray());
				}
			}
		}
		return colKeyPattens;
	}

	// Get all prefix strings of a column
	public Pair<ArrayList<String>[], ArrayList<Integer>[]> extractAllPrefixStringsOfColsSingleLine(boolean reverse) {
		ArrayList<String>[] prefixStrings = new ArrayList[ncols];
		ArrayList<Integer>[] rowIndexes = new ArrayList[ncols];
		for(int c=0; c< ncols; c++){
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
				String str = sampleRawIndexes.get(rowIndex).getSubString(0, mapCol[r][colIndex]);
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

	// Check the sequential list of keys are on a string
	private Integer getIndexOfKeysOnString(String str, ArrayList<String> key, int beginPos) {
		int currPos = beginPos;
		boolean flag = true;
		for(String k : key) {
			int index = str.indexOf(k, currPos);
			if(index != -1)
				currPos = index + k.length();
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

	private Pair<String, Integer> getPreviousLines(int beginLine, int endLine){
		StringBuilder sb = new StringBuilder();
		for(int i= Math.max(0, beginLine); i <= endLine; i++)
			sb.append(sampleRawIndexes.get(i).getRaw());
		String str = sb.toString();
		return new Pair<>(str, str.length() - sampleRawIndexes.get(endLine).getRawLength());
	}

	/////////////////////////////////////////////////////////////////////////////
	//                    Methods For Multi Lines Mapping                     //
	////////////////////////////////////////////////////////////////////////////
	// This implementation is for nested datasets are scattered on multiple lines
	// The following steps are required:
	// 1.  Extract all prefix strings per column
	// 2. Build key pattern tree for each column
	// 3. Build key pattern for end of values

	// Build key pattern tree for each column
	private KeyTrie[] buildColsKeyPatternMultiRow(){
		Pair<ArrayList<String>[], Pair<Integer, Integer>[]> prefixStrings = extractAllPrefixStringsOfColsMultiLine(true);
		ArrayList<String>[] suffixStrings = extractAllSuffixStringsOfColsMultiLine();

		KeyTrie[] colKeyPattens = new KeyTrie[ncols];
		for(int c=0; c<ncols; c++){
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
				for(ArrayList<String> keyPattern: keyPatterns) {
					boolean newCheck = checkKeyPatternIsUnique(prefixStrings.getKey()[c], keyPattern);
					check |= newCheck;
					if(newCheck){
						trie.setAPrefixPath(keyPattern);
					}
				}

				if(!check){
					remainedPrefixes.clear();
					boolean flag = true;
					for(ArrayList<String> keyPattern: keyPatterns){
						ArrayList<String> remainedPrefix = new ArrayList<>();
						for(String ps : prefixStrings.getKey()[c])
							remainedPrefix.add(getRemainedSubstring(ps, keyPattern));

						intersect = findStartWithIntersectOfStrings(remainedPrefix);
						if(intersect != null)
							trie.insertPrefixKeysConcurrent(intersect);
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
			for(String suffix: suffixStrings[c]) {
				trie.insertSuffixKeys(suffix.toCharArray());
			}
			colKeyPattens[c] = trie;
		}
		return colKeyPattens;
	}

	// Extract prefix strings:
	private Pair<ArrayList<String>[], Pair<Integer, Integer>[]> extractAllPrefixStringsOfColsMultiLine(boolean reverse){

		ArrayList<String>[] result = new ArrayList[ncols];
		Pair<Integer, Integer>[] minmax = new Pair[ncols];
		BitSet[] tmpUsedLines = new BitSet[nlines];
		BitSet[] usedLines = new BitSet[nlines];
		for(int r=0; r<nrows; r++)
			tmpUsedLines[r] = new BitSet();

		for(int r=0; r<nrows; r++)
			for(int c=0; c<ncols;c++)
				if(mapRow[r][c]!=-1)
					tmpUsedLines[r].set(mapRow[r][c]);

		for(int r=0; r<nrows; r++) {
			usedLines[r] = new BitSet(nlines);
			for(int i=0; i<nrows; i++) {
				if(i!=r)
					usedLines[r].or(tmpUsedLines[i]);
			}
		}

		// extract prefix strings
		for(int c = 0; c < ncols; c++){
			result[c] = new ArrayList<>();
			int min = 0;
			int max = 0;
			for(int r=0; r<nrows;r++){
				int rowIndex = mapRow[r][c];
				if(rowIndex == -1)
					continue;
				StringBuilder sb = new StringBuilder();
				int lastLine = 0;

				for(int i= rowIndex -1; i>=0; i--)
					if(usedLines[r].get(i)) {
						lastLine = i;
						break;
					}
				for(int i= lastLine; i<rowIndex; i++) {
					if(sampleRawIndexes.get(i).getRawLength() > 0 )
						sb.append(sampleRawIndexes.get(i).getRaw()).append("\n");
				}
				String str = sampleRawIndexes.get(rowIndex).getSubString(0, mapCol[r][c]);
				if(str.length() > 0 && !str.equals("\n"))
					sb.append(str);
				else if(lastLine < rowIndex)
					sb.deleteCharAt(sb.length()-1);


				if(reverse)
					result[c].add(sb.reverse().toString());
				else
					result[c].add(sb.toString());
				max = Math.max(max, sb.length());
				if(sb.length()< min || min == 0)
					min = sb.length();
				minmax[c] = new Pair<>(min, max);
			}
		}
		return new Pair<>(result, minmax);
	}

	private String findStartWithIntersectOfStrings(ArrayList<String> strList, int minLength){
		StringBuilder sb = new StringBuilder();
		int i = 0;
		boolean flag = true;
		do {
			char ch = strList.get(0).charAt(i);
			for(int j=1; j<Math.min(strList.size(), minLength); j++){
				char cch = strList.get(j).charAt(i);
				if(ch !=cch || ch=='\n'){
					flag = false;
					break;
				}
			}
			if(flag)
				sb.append(ch);
			i++;
		}while(flag && i< minLength);
		return sb.toString();

	}

	private HashSet<String> findStartWithIntersectOfStrings(ArrayList<String> strList){
		// 1. Extract all substrings
		// 2. Find intersection of substrings

		HashSet<String>[] substrings = new HashSet[strList.size()];
		for(int i=0; i< strList.size(); i++)
			substrings[i] = new HashSet<>();

		for(int w = windowSize; w > 2; w--) {
			for(int i=0; i<strList.size(); i++) {
				substrings[i].clear();
				substrings[i].addAll(getAllSubstringsOfAString(strList.get(i), w));
			}

			HashSet<String> totalIntersect = new HashSet<>(substrings[0]);
			for(int r=1; r<substrings.length; r++)
				totalIntersect.retainAll(substrings[r]);

			if(totalIntersect.size() > 0)
				return  totalIntersect;

		}
		return null;
	}

	private boolean checkKeyPatternIsUnique(ArrayList<String> prefixStrings, ArrayList<String> keys){
		for(String ps: prefixStrings){
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
			}while(true);
			if(patternCount!=1)
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
			if(startPos==-1)
				startPos = currPos;
		}
		if(flag)
			return new Pair<>(startPos, currPos+key.get(key.size()-1).length());
		else
			return new Pair<>(-1,-1);
	}

	private ArrayList<String> getAllSubstringsOfAString(String str,int size){
		ArrayList<String> result = new ArrayList<>();
		if(str == null)
			return  result;
		for(int i = 0; i <= str.length() - size; i++){
			String s = str.substring(i, i + size);
			if(!s.contains("\n"))
				result.add(s);
		}
		return result;
	}

	private String getRemainedSubstring(String str, ArrayList<String> keys){
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
