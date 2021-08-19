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

import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

public class ReaderMapping {

	private int[][] mapRow;
	private int[][] mapCol;
	private int[][] mapSize;
	private boolean symmetric;
	private boolean skewSymmetric;
	private boolean isUpperTriangular;
	private int skewCoefficient;

	private final ArrayList<String> sampleRawRows;
	private MatrixBlock sampleMatrix;
	private final boolean mapped;
	private final int nrows;
	private final int ncols;
	private long nnz;

	public ReaderMapping(String raw, MatrixBlock matrix) throws Exception {

		sampleMatrix = matrix;
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));

		nrows = matrix.getNumRows();
		ncols = matrix.getNumColumns();
		int nlines = 0;
		nnz = matrix.getNonZeros();

		NumberTrimFormat[][] NTF = convertMatrixTONumberTrimFormat(matrix);

		String value;
		sampleRawRows = new ArrayList<>();
		while((value = br.readLine()) != null) {
			sampleRawRows.add(value);
			nlines++;
		}

		// First Check for General Mapping
		boolean isMapped = findMapping(nlines, NTF);

		if(!isMapped) {

			// Symmetric and Skew-Symmetric check:
			symmetric = nrows == ncols;
			skewSymmetric = nrows == ncols;

			for(int r = 0; r < nrows; r++) {
				for(int c = 0; c < ncols; c++) {
					if(symmetric)
						symmetric = sampleMatrix.getValue(r, c) == sampleMatrix.getValue(r, c);
					if(symmetric) {
						if(r != c)
							skewSymmetric = sampleMatrix.getValue(r, c) == sampleMatrix.getValue(r, c) * (-1);
						else
							skewSymmetric = sampleMatrix.getValue(r, c) == 0;
					}
				}
			}

			if(symmetric) {

				// Lower Triangular
				isUpperTriangular = false;
				transferSampleMatrixTriangular(isUpperTriangular);
				isMapped = findMapping(nlines, NTF);

				// Upper Triangular
				if(!isMapped) {
					isUpperTriangular = true;
					sampleMatrix = matrix;
					transferSampleMatrixTriangular(isUpperTriangular);
					isMapped = findMapping(nlines, NTF);
				}
			}
			// Skew-Symmetric check:
			else if(skewSymmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				skewCoefficient = 1;
				transferSampleMatrixTriangular(isUpperTriangular);
				isMapped = findMapping(nlines, NTF);

				// Lower Triangular Skew
				if(!isMapped) {
					skewCoefficient = -1;
					skewSampleMatrix(skewCoefficient);
					NTF = convertMatrixTONumberTrimFormat(sampleMatrix);
					isMapped = findMapping(nlines, NTF);
				}

				// Upper Triangular
				if(!isMapped) {
					isUpperTriangular = true;
					skewCoefficient = 1;
					sampleMatrix = matrix;
					transferSampleMatrixTriangular(isUpperTriangular);
					NTF = convertMatrixTONumberTrimFormat(sampleMatrix);
					isMapped = findMapping(nlines, NTF);
				}
				// Upper Triangular Skew
				if(!isMapped) {
					skewCoefficient = -1;
					skewSampleMatrix(skewCoefficient);
					NTF = convertMatrixTONumberTrimFormat(sampleMatrix);
					isMapped = findMapping(nlines, NTF);
				}
			}
		}
		mapped = isMapped;
	}

	private boolean findMapping(int nlines, NumberTrimFormat[][] NTF) {

		mapRow = new int[nrows][ncols];
		mapCol = new int[nrows][ncols];
		mapSize = new int[nrows][ncols];

		// Set "-1" as default value for all defined matrix
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				mapRow[r][c] = mapCol[r][c] = mapSize[r][c] = -1;

		int itRow = 0;
		ArrayList<Integer> colIndexes = new ArrayList<>();
		ArrayList<Integer> colIndexSizes = new ArrayList<>();
		for(int r = 0; r < nrows; r++) {

			NumberTrimFormat[] ntfRow = new NumberTrimFormat[ncols];
			for(int i = 0; i < ncols; i++) {
				ntfRow[i] = NTF[r][i].getACopy();
			}
			Arrays.sort(ntfRow);

			for(NumberTrimFormat ntf : ntfRow) {
				if(ntf.actualValue == 0) {
					continue;
				}
				int c = ntf.c;
				while(itRow < nlines) {
					String row = sampleRawRows.get(itRow);
					NumberMappingInfo nmi = getCellMapping(row, ntf, colIndexes, colIndexSizes);
					if(nmi.mapped) {
						mapRow[r][c] = itRow;
						mapCol[r][c] = nmi.index;
						mapSize[r][c] = nmi.size;
						colIndexes.add(nmi.index);
						colIndexSizes.add(nmi.size);
						break;
					}
					else {
						itRow++;
						colIndexes.clear();
						colIndexSizes.clear();
					}
				}
			}
		}
		boolean flagMap = true;
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] == -1 && NTF[r][c].actualValue != 0)
					flagMap = false;
		return flagMap;
	}

	private void transferSampleMatrixTriangular(boolean isUpper) throws Exception {
		if(nrows != ncols)
			throw new Exception("For upper triangular matrix both row and col should be same!");

		for(int r = 0; r < nrows; r++) {
			if(isUpper) {
				for(int c = 0; c < r - 1; c++) {
					sampleMatrix.setValue(r, c, 0);
				}
			}
			else {
				for(int c = r + 1; c < ncols; c++) {
					sampleMatrix.setValue(r, c, 0);
				}
			}
		}
		nnz = sampleMatrix.getNonZeros();
	}

	private void skewSampleMatrix(int coefficient) throws Exception {

		if(coefficient != 1 && coefficient != -1)
			throw new Exception("The value of Coefficient have to be 1 or -1!");

		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				sampleMatrix.setValue(r, r, sampleMatrix.getValue(r, c) * coefficient);
	}

	// Convert: convert each value of a sample matrix to NumberTrimFormat
	private static NumberTrimFormat[][] convertMatrixTONumberTrimFormat(MatrixBlock matrix) {
		int nrows = matrix.getNumRows();
		int ncols = matrix.getNumColumns();
		NumberTrimFormat[][] result = new NumberTrimFormat[nrows][ncols];
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++) {
				result[r][c] = new NumberTrimFormat(c, matrix.getValue(r, c));
			}
		return result;
	}

	// Looking for a double(Number Trim Format) on a String
	public NumberMappingInfo getCellMapping(String row, NumberTrimFormat ntf, ArrayList<Integer> reservedIndex,
		ArrayList<Integer> reservedSize) {

		NumberMappingInfo result = null;
		ArrayList<String> rowChunks = new ArrayList<>();
		ArrayList<Integer> baseIndexes = new ArrayList<>();

		BitSet bitSet=new BitSet(row.length());
		for(int i=0;i<reservedIndex.size();i++) {
			bitSet.set(reservedIndex.get(i), reservedIndex.get(i)+reservedSize.get(i));
		}
		getChunksOFString(row,bitSet,rowChunks,baseIndexes);

		int rci = -1;
		for(String rc : rowChunks) {
			rci++;
			result = ntf.getMappingInfo(rc);
			if(result.mapped)
				break;
		}
		if(result == null)
			result = new NumberMappingInfo();

		if(result.mapped) {
			result.index += baseIndexes.get(rci);
		}
		return result;
	}

	public final FileFormatProperties getFormatProperties() throws Exception {
		FileFormatProperties ffp;
		if(isRowRegular()) {
			ffp = getFileFormatPropertiesOfRRCRMapping();

			if(ffp == null) {
				ffp = getFileFormatPropertiesOfRRCIMapping();
			}
			return ffp;
		}
		else {

			FileFormatPropertiesMM.MMFormat format;
			if(sampleRawRows.size() == nnz) {
				format = FileFormatPropertiesMM.MMFormat.COORDINATE;
			}
			else
				format = FileFormatPropertiesMM.MMFormat.ARRAY;

			FileFormatPropertiesMM.MMField field = FileFormatPropertiesMM.MMField.REAL;
			for(int r = 0; r < nrows && field == FileFormatPropertiesMM.MMField.REAL; r++) {
				for(int c = 0; c < ncols; c++) {
					if(sampleMatrix.getValue(r, c) != (int) sampleMatrix.getValue(r, c)) {
						field = FileFormatPropertiesMM.MMField.INTEGER;
						break;
					}
				}
			}
			FileFormatPropertiesMM.MMSymmetry symmetry;
			if(symmetric)
				symmetry = FileFormatPropertiesMM.MMSymmetry.SYMMETRIC;
			else if(skewSymmetric)
				symmetry = FileFormatPropertiesMM.MMSymmetry.SKEW_SYMMETRIC;
			else
				symmetry = FileFormatPropertiesMM.MMSymmetry.GENERAL;

			String delim = getDelimsOfRIMapping();
			if(delim != null) {
				ffp = new FileFormatPropertiesMM(format, field, symmetry);
				return ffp;
			}
			else
				return null;
		}
	}

	public final boolean isRowRegular() {
		int nrows = mapRow.length;
		int ncols = mapRow[0].length;
		boolean result = true;
		int rValue = -1;
		for(int c = 0; c < ncols; c++) {
			if(mapRow[0][c] != -1) {
				rValue = mapRow[0][c];
				break;
			}
		}

		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(mapRow[r][c] != -1 && mapRow[r][c] != rValue + r) {
					result = false;
					break;
				}
			}
		}
		return result;
	}

	/* Get delimiters between two indexes.
		 Row String:     1,2,3,4,5
		 Sample Matrix: [1 2 3 4 5 ]
		 Map Col:       [0 2 4 6 8 ]
		 result:        ["," "," "," "," ","]
		*/
	public final FileFormatProperties getFileFormatPropertiesOfRRCRMapping() {

		int nrows = mapRow.length;
		int ncols = mapRow[0].length;

		String[][] delims = new String[nrows][ncols + 1];
		for(int r = 0; r < nrows; r++) {
			int[] rCol = new int[ncols];
			System.arraycopy(mapCol[r], 0, rCol, 0, ncols);
			String row = sampleRawRows.get(r);

			Map<Integer, Integer> ciMap = new HashMap<>();
			Map<Integer, Integer> icMap = new HashMap<>();
			Map<Integer, String> idMap = new HashMap<>();
			for(int i = 0; i < ncols; i++) {
				if(rCol[i] == -1)
					continue;
				ciMap.put(i, rCol[i]);
				icMap.put(rCol[i], i);
			}

			boolean remindRowString = false;
			if(icMap.get(row.length() - 1) == null) {
				ciMap.put(ncols, row.length() - 1);
				icMap.put(row.length() - 1, ncols);
				remindRowString = true;
			}
			Arrays.sort(rCol);
			int sIndex = 0;
			int eIndex;
			String sValue = "";
			double value = 0;
			int colIndex = -1;
			for(Integer i : rCol) {
				if(i == -1) {
					continue;
				}
				eIndex = i;
				String subRow = row.substring(sIndex, eIndex);

				if(sIndex != 0 && subRow.length() > 1) {
					int mergeCount = mergeDelimiters(value, sValue, subRow);
					if(mergeCount > 0) {
						mapSize[r][colIndex] += mergeCount;
						subRow = subRow.substring(mergeCount);
					}
				}

				colIndex = icMap.get(i);
				sValue = row.substring(mapCol[r][colIndex], mapCol[r][colIndex] + mapSize[r][colIndex]).toLowerCase();
				value = Double.parseDouble(sValue);
				idMap.put(i, subRow);
				sIndex = eIndex + mapSize[r][colIndex];
			}
			if(remindRowString) {
				String subRow = row.substring(sIndex);
				int mergeCount = mergeDelimiters(value, sValue, subRow);
				if(mergeCount > 0) {
					mapSize[r][colIndex] += mergeCount;
					subRow = subRow.substring(mergeCount);
				}
				idMap.put(row.length() - 1, subRow);
			}

			for(int c = 0; c <= ncols; c++) {
				Integer cim = ciMap.get(c);
				String d = "";
				if(cim != null)
					d = idMap.get(cim);
				delims[r][c] = d;
			}
		}

		ArrayList<String> rowDelims = new ArrayList<>();
		HashSet<String> naString = new HashSet<>();
		int maxSizeOfToken = 0;
		String stringToken = null;

		// append all delimiters as a string and then tokenize it
		for(int r = 0; r < nrows; r++) {
			StringBuilder sbRow = new StringBuilder();
			for(int c = 0; c < ncols + 1; c++) {
				sbRow.append(delims[r][c]);
				if(maxSizeOfToken == 0 || (delims[r][c].length() > 0 && delims[r][c].length() < maxSizeOfToken)) {
					maxSizeOfToken = delims[r][c].length();
					stringToken = delims[r][c];
				}
			}
			rowDelims.add(sbRow.toString());
		}

		String uniqueDelimiter = null;
		StringBuilder token = new StringBuilder();

		while(token.length() < maxSizeOfToken) {
			token.append(stringToken.charAt(token.length()));

			boolean flagCurrToken = true;
			HashSet<String> ns = new HashSet<>();
			for(int r = 0; r < nrows; r++) {
				int rowDelimCount = 0;
				String row = rowDelims.get(r);
				int sPos = 0;
				do {
					int index = row.substring(sPos).indexOf(token.toString());
					if(index != -1) {
						rowDelimCount++;
						String nv = row.substring(sPos, sPos + index);
						if(nv.length() > 0)
							ns.add(nv);
						sPos += index + token.length();
					}
					else
						break;
				}
				while(sPos <= row.length());

				if(rowDelimCount != ncols - 1) {
					flagCurrToken = false;
					break;
				}
			}
			if(flagCurrToken) {
				uniqueDelimiter = token.toString();
				naString = ns;
			}

		}
		if(uniqueDelimiter != null) {
			FileFormatPropertiesCSV ffpcsv = new FileFormatPropertiesCSV(false, uniqueDelimiter, false);
			ffpcsv.setNAStrings(naString);
			ffpcsv.setDescription("CSV Format Recognized");
			return ffpcsv;
		}
		else
			return null;
	}

	/*
	 In some case we need to merge delimiter chars with value
	 Example:
	 SampleMatrix=[1   2   3   ]
	 SampleRaw  =[ 1.0,2.0,3.0 ]
	 Delimiters =[ ".0," ".0," ".0,"]
	 So ".0" values have to merge to SampleMatrix values
	 Final Delimiters = ["," "," ","]
	*/
	private int mergeDelimiters(double value, String stringValue, String delimiters) {
		StringBuilder sb = new StringBuilder();
		sb.append(stringValue);
		boolean flagE = stringValue.contains("e");
		boolean flagD = stringValue.contains(".");
		boolean flagS = false;
		int mergeCount = 0;

		for(Character c : delimiters.toCharArray()) {
			if(Character.isDigit(
				c) || (!flagD && c == '.') || (!flagE && (c == 'e' || c == 'E')) || (flagE && !flagS && (c == '+' || c == '-'))) {

				sb.append(c);

				if(c == '.')
					flagD = true;

				if(c == 'e' || c == 'E') {
					flagE = true;
				}
				else if(c == '+' || c == '-')
					flagS = true;
				else {
					if(Double.parseDouble(sb.toString()) == value) {
						mergeCount = sb.length() - stringValue.length();
					}
//					else
//						break;
				}
			}
			else
				break;
		}
		return mergeCount;
	}

	private String getDelimsOfRIMapping() throws Exception {

		// FirstRowIndex = 0, FirstColIndex = 0
		String delims = getDelimsOfMapping(0, 0);

		// FirstRowIndex = 1, FirstColIndex = 1
		if(delims == null)
			delims = getDelimsOfMapping(1, 1);

		// FirstRowIndex = 1, FirstColIndex = 0
		if(delims == null)
			delims = getDelimsOfMapping(1, 0);

		// FirstRowIndex = 0, FirstColIndex = 1
		if(delims == null)
			delims = getDelimsOfMapping(0, 1);

		return delims;
	}

	private String getDelimsOfMapping(int firstRowIndex, int firstColIndex) throws Exception {

		int nLines = sampleRawRows.size();
		int nrows = sampleMatrix.getNumRows();
		int ncols = sampleMatrix.getNumColumns();
		Set<Integer> checkedRow = new HashSet<>();
		RowColValue[] mapRCV = new RowColValue[(int) nnz];

		boolean rcvMapped = false;
		for(int p = 0; p < 4 && !rcvMapped; p++) {
			checkedRow.clear();
			int nzIndex = 0;
			rcvMapped = true;
			for(int r = nrows - 1; r >= 0 && rcvMapped; r--) {
				for(int c = ncols - 1; c >= 0 && rcvMapped; c--) {
					RowColValue rcv = new RowColValue(p, r + firstRowIndex, c + firstColIndex,
						sampleMatrix.getValue(r, c));
					int index = 0;
					do {
						if(checkedRow.contains(index))
							continue;
						String row = sampleRawRows.get(index);
						if(rcv.isMapped(row)) {
							checkedRow.add(index);
							mapRCV[nzIndex++] = rcv;
						}
					}
					while(++index < nLines && !rcv.isMapped());
					rcvMapped = rcv.isMapped();
				}
			}
		}
		// All combinations were wrong
		if(!rcvMapped) {
			return null;
		}
		else {
			HashSet<String> delims = new HashSet<>();
			mapRCV[0].findDelims();
			delims.add(mapRCV[0].v1Delim);
			delims.add(mapRCV[0].v2Delim);
			int minDelimLength = Math.min(mapRCV[0].v1Delim.length(), mapRCV[0].v2Delim.length());
			for(int i = 1; i < nnz; i++) {
				mapRCV[i].findDelims();
				delims.add(mapRCV[i].v1Delim);
				delims.add(mapRCV[i].v2Delim);
			}

			HashSet<String> token = null;
			for(int l = 1; l < minDelimLength; l++) {
				boolean flagToken = true;
				token = new HashSet<>();
				while(delims.iterator().hasNext()) {
					String delim = delims.iterator().next();
					if(delim.length() % l != 0) {
						flagToken = false;
						break;
					}
					for(int i = 0; i < delim.length() - l; i += l)
						token.add(delim.substring(i, i + l));
					if(token.size() > 1) {
						flagToken = false;
						break;
					}
				}
				if(flagToken)
					break;
			}
			if(token != null) {
				return token.iterator().next();
			}
			else
				return null;
		}
	}

	public FileFormatPropertiesLIBSVM getFileFormatPropertiesOfRRCIMapping() {

		FileFormatPropertiesLIBSVM ffplibsvm;
		int firstColIndex = 0;

		//FirstColIndex = 0
		ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);

		//FirstColIndex = 1
		if(ffplibsvm == null) {
			firstColIndex = 1;
			ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);
		}

		if(ffplibsvm != null)
			ffplibsvm.setDescription("LibSVM Format Recognized: First Index Started From " + firstColIndex);

		return ffplibsvm;
	}

	private FileFormatPropertiesLIBSVM getDelimsOfRRCIMapping(int firstColIndex) {

		Map<String, Set<String>> tokens = new HashMap<>();
		Set<String> allTokens = new HashSet<>();

		for(int c = ncols - 1; c >= 0; c--) {
			double v = sampleMatrix.getValue(0, c);
			if(v == 0)
				continue;

			String key = (c + firstColIndex) + "," + v;
			Set<String> token = tokens.get(key);

			ColIndexValue civ = new ColIndexValue(c + firstColIndex, v);
			String row = sampleRawRows.get(mapRow[0][c]);

			if(token == null) {
				token = new HashSet<>();
				tokens.put(key, token);
			}
			token.addAll(civ.getMappedTokens(row));
			allTokens.addAll(token);
		}

		ArrayList<String> missedKeys = new ArrayList<>();
		ArrayList<Integer> labelIndex = new ArrayList<>();
		ArrayList<String> selectedTokens = new ArrayList<>();

		for(String key : tokens.keySet()) {
			if(tokens.get(key).size() == 0)
				missedKeys.add(key);
		}
		if(missedKeys.size() > 1)
			return null;
		else {
			for(String t : allTokens) {
				missedKeys.clear();
				for(String key : tokens.keySet()) {
					if(!tokens.get(key).contains(t)) {
						missedKeys.add(key);
					}
				}
				if(missedKeys.size() == 1) {
					int li = Integer.parseInt(missedKeys.iterator().next().split(",")[0]);
					labelIndex.add(li);
					selectedTokens.add(t);
				}
			}
		}
		String separator = null;
		String indexSeparator = null;

		boolean isVerify = false;
		for(int i = 0; i < selectedTokens.size() && !isVerify; i++) {
			isVerify = true;
			indexSeparator = selectedTokens.get(i);
			for(int r = 0; r < nrows; r++) {
				separator = verifyRowWithIndexDelim(r, indexSeparator , labelIndex.get(i), firstColIndex);
				if(separator == null) {
					isVerify = false;
					break;
				}
			}
		}
		if(isVerify) {
			return new FileFormatPropertiesLIBSVM(separator, indexSeparator);
		}
		else
			return null;
	}

	private String verifyRowWithIndexDelim(int rowID, String indexDelim, int labelIndex, int firstColIndex) {
		String row = sampleRawRows.get(rowID);
		int length = row.length();
		BitSet bitSet = new BitSet(length);
		ArrayList<String> stringChunks = new ArrayList<>();
		ArrayList<Integer> baseIndexes = new ArrayList<>();

		for(int c = ncols - 1; c >= 0; c--) {
			if(mapRow[rowID][c] != -1) {
				NumberTrimFormat ntfColIndex = new NumberTrimFormat(c + firstColIndex);
				NumberTrimFormat ntfColValue = new NumberTrimFormat(sampleMatrix.getValue(rowID, c));

				stringChunks.clear();
				baseIndexes.clear();
				getChunksOFString(row, bitSet, stringChunks, baseIndexes);
				boolean itemVerify = false;

				for(int i = 0; i < stringChunks.size() && !itemVerify; i++) {
					String chunk = stringChunks.get(i);
					int sPosition = 0;
					while(sPosition < chunk.length()) {
						String subChunk = chunk.substring(sPosition);

						NumberMappingInfo nmiIndex;
						if(labelIndex - firstColIndex != c) {

							// check Index text
							nmiIndex = ntfColIndex.actualValue == 0 ? ntfColIndex
								.getMappingInfoIncludeZero(subChunk) : ntfColIndex.getMappingInfo(subChunk);

							if(!nmiIndex.mapped)
								break;

							nmiIndex.size += mergeDelimiters(ntfColIndex.actualValue,
								subChunk.substring(nmiIndex.index, nmiIndex.index + nmiIndex.size),
								subChunk.substring(nmiIndex.index + nmiIndex.size));

							subChunk = subChunk.substring(nmiIndex.index + nmiIndex.size);

							// check the delimiter text
							if(indexDelim.length() > subChunk.length()) {
								sPosition++;
								continue;
							}

							String chunkDelim = subChunk.substring(0, indexDelim.length());

							if(!indexDelim.equals(chunkDelim)) {
								sPosition++;
								continue;
							}
							subChunk = subChunk.substring(indexDelim.length());
						}
						else {
							nmiIndex = new NumberMappingInfo();
							nmiIndex.index = 0;
							nmiIndex.size = 0;
							nmiIndex.mapped = true;
						}

						// check the value text
						NumberMappingInfo nmiValue = ntfColValue.getMappingInfo(subChunk);
						nmiValue.size += mergeDelimiters(ntfColValue.actualValue,
							subChunk.substring(nmiValue.index, nmiValue.index + nmiValue.size),
							subChunk.substring(nmiValue.index + nmiValue.size));

						if(nmiValue.mapped) {
							itemVerify = true;
							nmiIndex.index += sPosition + baseIndexes.get(i);
							if(labelIndex - firstColIndex != c) {
								nmiValue.index += indexDelim.length();
							}
							nmiValue.index += nmiIndex.index + nmiIndex.size;
							bitSet.set(nmiIndex.index, nmiValue.index + nmiValue.size);
							break;
						}
						sPosition++;
					}
				}

				if(!itemVerify)
					return null;
			}
		}

		stringChunks.clear();
		baseIndexes.clear();
		getChunksOFString(row, bitSet, stringChunks, baseIndexes);
		Set<String> separators = new HashSet<>();
		separators.addAll(stringChunks);
		String separator = separators.size() == 1 ? separators.iterator().next() : null;

		if(separator == null)
			return null;

		return separator;
	}

	private void getChunksOFString(String row, BitSet bitSet, ArrayList<String> stringChunks,
		ArrayList<Integer> baseIndexes) {

		int length = row.length();
		int sIndex, eIndex;
		for(int i = 0; i < length;) {
			// skip all reserved indexes
			for(int j = i; j < length; j++) {
				if(bitSet.get(j))
					i++;
				else
					break;
			}
			sIndex = i;
			// Extract unreserved sub text
			for(int j = i; j < length; j++) {
				if(!bitSet.get(j))
					i++;
				else
					break;
			}
			eIndex = i;
			String subRow = row.substring(sIndex, eIndex);

			if(subRow.length() > 0) {
				stringChunks.add(subRow);
				baseIndexes.add(sIndex);
			}
		}
	}

	class RowColValue {
		private final NumberTrimFormat ntfV0;
		private final NumberTrimFormat ntfV1;
		private final NumberTrimFormat ntfV2;
		private NumberMappingInfo nmiV0;
		private NumberMappingInfo nmiV1;
		private NumberMappingInfo nmiV2;
		private boolean mapped;
		private int type;
		private String rowString;

		private String v0Delim;
		private String v1Delim;
		private String v2Delim;

		public RowColValue(int type, int row, int col, double value) throws Exception {
			this.type = type;
			rowString = null;
			switch(type) {
				//0 : Row, Col, Value
				case 0:
					ntfV0 = new NumberTrimFormat(row);
					ntfV1 = new NumberTrimFormat(col);
					ntfV2 = new NumberTrimFormat(value);
					break;
				//1 : Col, Row, Value
				case 1:
					ntfV0 = new NumberTrimFormat(col);
					ntfV1 = new NumberTrimFormat(row);
					ntfV2 = new NumberTrimFormat(value);
					break;
				//2 : Value, Row, Col
				case 2:
					ntfV0 = new NumberTrimFormat(value);
					ntfV1 = new NumberTrimFormat(row);
					ntfV2 = new NumberTrimFormat(col);
					break;
				//3 : Value, Col, Row
				case 3:
					ntfV0 = new NumberTrimFormat(value);
					ntfV1 = new NumberTrimFormat(col);
					ntfV2 = new NumberTrimFormat(row);
					break;
				default:
					throw new Exception("The Auto Generate Reader just supporting 4 types(type value can be 0 to 4)!!");
			}
			mapped = false;
		}

		public boolean isMapped(String row) {

			byte hasZero = 0b000;
			if(ntfV0.actualValue == 0)
				hasZero |= 0b100;

			if(ntfV1.actualValue == 0)
				hasZero |= 0b010;

			if(ntfV2.actualValue == 0)
				hasZero |= 0b001;

			switch(hasZero) {
				//010
				case 2:
					nmiV0 = ntfV0.getMappingInfo(row);
					if(!nmiV0.mapped)
						return false;

					nmiV2 = ntfV2.getMappingInfo(row.substring(nmiV0.index + nmiV0.size));
					if(!nmiV2.mapped)
						return false;
					nmiV2.index += nmiV0.index + nmiV0.size;

					nmiV1 = ntfV1.getMappingInfo(row.substring(nmiV0.index + nmiV0.size, nmiV2.index));
					if(!nmiV1.mapped)
						return false;
					nmiV1.index += nmiV0.index + nmiV0.size;
					break;

				//100 , 101
				case 4:
				case 5:
					nmiV1 = ntfV1.getMappingInfo(row);
					if(!nmiV1.mapped)
						return false;

					nmiV2 = ntfV2.getMappingInfoIncludeZero(row.substring(nmiV1.index + nmiV1.size));
					if(!nmiV2.mapped)
						return false;
					nmiV2.index += nmiV1.index + nmiV1.size;

					nmiV0 = ntfV0.getMappingInfoIncludeZero(row.substring(0, nmiV1.index));
					if(!nmiV0.mapped)
						return false;
					break;

				// 110
				case 6:
					nmiV2 = ntfV2.getMappingInfo(row);
					if(!nmiV2.mapped)
						return false;

					nmiV0 = ntfV0.getMappingInfoIncludeZero(row.substring(0, nmiV2.index));
					if(!nmiV0.mapped)
						return false;

					nmiV1 = ntfV1.getMappingInfoIncludeZero(row.substring(nmiV0.index + nmiV0.size));
					if(!nmiV1.mapped)
						return false;
					nmiV1.index += nmiV0.index + nmiV0.size;
					break;

				default:
					nmiV0 = ntfV0.getMappingInfo(row);
					if(!nmiV0.mapped)
						return false;

					nmiV1 = ntfV1.actualValue == 0 ? ntfV1
						.getMappingInfoIncludeZero(row.substring(nmiV0.index + nmiV0.size)) : ntfV1
						.getMappingInfo(row.substring(nmiV0.index + nmiV0.size));
					if(!nmiV1.mapped)
						return false;
					nmiV1.index += nmiV0.index + nmiV0.size;

					nmiV2 = ntfV2.actualValue == 0 ? ntfV2
						.getMappingInfoIncludeZero(row.substring(nmiV1.index + nmiV1.size)) : ntfV2
						.getMappingInfo(row.substring(nmiV1.index + nmiV1.size));
					if(!nmiV2.mapped)
						return false;
					nmiV2.index += nmiV1.index + nmiV1.size;
					break;
			}
			mapped = true;
			rowString = row;
			return true;
		}

		public boolean isMapped() {
			return mapped;
		}

		public void findDelims() throws Exception {
			if(!isMapped())
				throw new Exception("The values didn't match !!");

			v0Delim = rowString.substring(0, nmiV0.index);
			nmiV0.size += mergeDelimiters(ntfV0.actualValue, rowString.substring(nmiV0.index, nmiV0.index + nmiV0.size),
				rowString.substring(nmiV0.index + nmiV0.size));

			v1Delim = rowString.substring(nmiV0.index + nmiV0.size, nmiV1.index);
			nmiV1.size += mergeDelimiters(ntfV1.actualValue, rowString.substring(nmiV1.index, nmiV1.index + nmiV1.size),
				rowString.substring(nmiV1.index + nmiV1.size));

			v2Delim = rowString.substring(nmiV1.index + nmiV1.size, nmiV2.index);
			nmiV2.size += mergeDelimiters(ntfV2.actualValue, rowString.substring(nmiV2.index, nmiV2.index + nmiV2.size),
				rowString.substring(nmiV2.index + nmiV2.size));
		}

		public NumberMappingInfo getNmiV0() {
			return nmiV0;
		}

		public NumberMappingInfo getNmiV1() {
			return nmiV1;
		}

		public NumberMappingInfo getNmiV2() {
			return nmiV2;
		}

		public int getType() {
			return type;
		}

		public String getV0Delim() {
			return v0Delim;
		}

		public String getV1Delim() {
			return v1Delim;
		}

		public String getV2Delim() {
			return v2Delim;
		}
	}

	class ColIndexValue {
		private final NumberTrimFormat ntfColIndex;
		private final NumberTrimFormat ntfColValue;
		private NumberMappingInfo nmiColIndex;
		private NumberMappingInfo nmiColValue;
		private boolean mapped;
		private String indSep;
		private String separator;

		public ColIndexValue(int col, double value) {
			ntfColIndex = new NumberTrimFormat(col);
			ntfColValue = new NumberTrimFormat(value);
			indSep = null;
			separator = null;
			mapped = false;
			nmiColIndex = null;
			nmiColValue = null;
		}

		public ColIndexValue(NumberTrimFormat ntfColIndex, NumberTrimFormat ntfColValue, NumberMappingInfo nmiColIndex,
			NumberMappingInfo nmiColValue, String indSep) {
			this.ntfColIndex = ntfColIndex;
			this.ntfColValue = ntfColValue;
			this.nmiColIndex = nmiColIndex;
			this.nmiColValue = nmiColValue;
			this.indSep = indSep;
		}

		public Set<String> getMappedTokens(String row) {

			int sPosition = 0;
			Set<String> tokens = new HashSet<>();

			while(sPosition < row.length()) {
				NumberMappingInfo nmiIndex;

				nmiIndex = ntfColIndex.actualValue == 0 ? ntfColIndex
					.getMappingInfoIncludeZero(row.substring(sPosition)) : ntfColIndex
					.getMappingInfo(row.substring(sPosition));
				if(!nmiIndex.mapped) {
					break;
				}
				nmiIndex.index += sPosition;

				nmiIndex.size += mergeDelimiters(ntfColIndex.actualValue,
					row.substring(+nmiIndex.index, nmiIndex.index + nmiIndex.size),
					row.substring(nmiIndex.index + nmiIndex.size));

				NumberMappingInfo nmiValue = ntfColValue.getMappingInfo(row.substring(nmiIndex.index + nmiIndex.size));
				if(!nmiValue.mapped) {
					break;
				}
				nmiValue.index += nmiIndex.index + nmiIndex.size;

				nmiValue.size += mergeDelimiters(ntfColValue.actualValue,
					row.substring(nmiValue.index, nmiValue.index + nmiValue.size),
					row.substring(nmiValue.index + nmiValue.size));

				String t = row.substring(nmiIndex.index + nmiIndex.size, nmiValue.index);
				if(t.length() > 0)
					tokens.add(t);

				sPosition = nmiIndex.index + 1;
			}
			return tokens;
		}

		public void setIndSep(String indSep) {
			this.indSep = indSep;
		}

		public void setSeparator(String separator) {
			this.separator = separator;
		}

		public String getIndSep() {
			return indSep;
		}

		public String getSeparator() {
			return separator;
		}
	}

	public int[][] getMapRow() {
		return mapRow;
	}

	public int[][] getMapCol() {
		return mapCol;
	}

	public int[][] getMapSize() {
		return mapSize;
	}

	public boolean isSymmetric() {
		return symmetric;
	}

	public boolean isMapped() {
		return mapped;
	}

	public long getNnz() {
		return nnz;
	}
}
