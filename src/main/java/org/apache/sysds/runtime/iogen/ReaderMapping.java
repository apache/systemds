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

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Arrays;
import java.util.Iterator;

public abstract class ReaderMapping {

	protected int[][] mapRow;
	protected int[][] mapCol;
	protected int[][] mapSize;
	protected boolean symmetric;
	protected boolean skewSymmetric;
	protected boolean isUpperTriangular;
	protected int skewCoefficient;
	protected final ArrayList<String> sampleRawRows;

	protected boolean mapped;
	protected static int nrows;
	protected static int ncols;
	protected final int nlines;
	protected static long nnz;

	protected ValueTrimFormat[][] VTF;

	public ReaderMapping(String raw) throws Exception {
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		int nlines = 0;
		sampleRawRows = new ArrayList<>();
		while((value = br.readLine()) != null) {
			sampleRawRows.add(value);
			nlines++;
		}
		this.nlines = nlines;
		System.out.println("raw_nrows:" + sampleRawRows.size());
	}

	protected abstract void transferSampleTriangular(boolean isUpper) throws Exception;

	protected abstract void transferSampleSkew(int coefficient) throws Exception;

	protected abstract boolean isSchemaNumeric();

	protected abstract void cloneSample();

	protected abstract void retrieveSample();

	protected abstract ValueTrimFormat[][] convertSampleTOValueTrimFormat();

	// Matrix Reader Mapping
	public static class MatrixReaderMapping extends ReaderMapping {

		private MatrixBlock sampleMatrix;
		private MatrixBlock sampleMatrixClone;

		public MatrixReaderMapping(String raw, MatrixBlock matrix) throws Exception {
			super(raw);
			this.sampleMatrix = matrix;
			nrows = sampleMatrix.getNumRows();
			ncols = sampleMatrix.getNumColumns();
			nnz = sampleMatrix.getNonZeros();
			VTF = convertSampleTOValueTrimFormat();
			runMapping();
		}

		// Convert: convert each value of a sample matrix to NumberTrimFormat
		@Override
		protected ValueTrimFormat.NumberTrimFormat[][] convertSampleTOValueTrimFormat() {
			ValueTrimFormat.NumberTrimFormat[][] result = new ValueTrimFormat.NumberTrimFormat[nrows][ncols];
			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++) {
					result[r][c] = new ValueTrimFormat.NumberTrimFormat(c, sampleMatrix.getValue(r, c));
				}
			return result;
		}

		@Override protected void transferSampleTriangular(boolean isUpper) throws Exception {
			if(nrows != ncols)
				throw new Exception("For upper triangular both Row and Col should be same!");

			for(int r = 0; r < nrows; r++) {
				if(isUpper) {
					for(int c = 0; c < r; c++) {
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

		@Override protected void transferSampleSkew(int coefficient) throws Exception {
			if(coefficient != 1 && coefficient != -1)
				throw new Exception("The value of Coefficient have to be 1 or -1!");

			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++)
					sampleMatrix.setValue(r, c, sampleMatrix.getValue(r, c) * coefficient);
		}

		@Override protected boolean isSchemaNumeric() {
			return true;
		}

		@Override protected void cloneSample() {
			// Clone Sample Matrix
			sampleMatrixClone = new MatrixBlock(nrows, ncols, false);
			this.sampleMatrix.copy(0, nrows, 0, ncols, sampleMatrixClone, false);
		}

		@Override protected void retrieveSample() {
			// Retrieve SampleMatrix From Cloned Data
			this.sampleMatrixClone.copy(0, nrows, 0, ncols, sampleMatrix, false);
		}
	}

	// Frame Reader Mapping
	public static class FrameReaderMapping extends ReaderMapping {

		private FrameBlock sampleFrame;
		private FrameBlock sampleFrameClone;
		private Types.ValueType[] schema;
		private String[] names;

		public FrameReaderMapping(String raw, FrameBlock frame) throws Exception {
			super(raw);
			this.sampleFrame = frame;
			nrows = sampleFrame.getNumRows();
			ncols = sampleFrame.getNumColumns();
			schema = sampleFrame.getSchema();
			names = sampleFrame.getColumnNames();
			VTF = convertSampleTOValueTrimFormat();
			//TODO: set NNZ for Frame !!??
			runMapping();
		}

		// Convert: convert each value of a sample Frame to ValueTrimFormat(Number, String, and Boolean)
		@Override
		protected ValueTrimFormat[][] convertSampleTOValueTrimFormat() {
			ValueTrimFormat[][] result = new ValueTrimFormat[nrows][ncols];
			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++) {
					result[r][c] = ValueTrimFormat.createNewTrimFormat(schema[c], sampleFrame.get(r, c));
				}
			return result;
		}

		@Override protected boolean isSchemaNumeric() {
			boolean result = true;
			for(Types.ValueType vt : schema)
				result &= vt.isNumeric();
			return result;
		}

		@Override protected void transferSampleTriangular(boolean isUpper) throws Exception {
			if(nrows != ncols)
				throw new Exception("For upper triangular both Row and Col should be same!");

			for(int r = 0; r < nrows; r++) {
				if(isUpper) {
					for(int c = 0; c < r; c++) {
						sampleFrame.set(r, c, 0);
					}
				}
				else {
					for(int c = r + 1; c < ncols; c++) {
						sampleFrame.set(r, c, 0);
					}
				}
			}
		}

		@Override protected void transferSampleSkew(int coefficient) throws Exception {
			if(coefficient != 1 && coefficient != -1)
				throw new Exception("The value of Coefficient have to be 1 or -1!");

			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++)
					sampleFrame.set(r, c, (Double) sampleFrame.get(r, c) * coefficient);
		}

		@Override protected void cloneSample() {
			// Clone SampleFrame
			sampleFrameClone = new FrameBlock(schema, names);
			sampleFrameClone.ensureAllocatedColumns(nrows);
			sampleFrame.copy(0, nrows, 0, ncols, sampleFrameClone);
		}

		@Override protected void retrieveSample() {
			// Retrieve SampleFrame from Cloned Data
			sampleFrameClone.copy(0, nrows, 0, ncols, sampleFrame);
		}
	}

	public void runMapping() throws Exception {

		boolean isMapped = findMapping();
		boolean schemaNumeric = isSchemaNumeric();
		if(!isMapped ) {
			// Clone Sample Matrix/Frame
			cloneSample();

			// Symmetric and Skew-Symmetric check:
			symmetric = nrows == ncols;
			skewSymmetric = nrows == ncols && schemaNumeric;

			for(int r = 0; r < nrows; r++) {
				for(int c = 0; c < ncols; c++) {
					if(symmetric)
						symmetric = VTF[r][c].isEqual(VTF[c][r]);

					if(skewSymmetric) {
						if(r != c)
							skewSymmetric = ((ValueTrimFormat.NumberTrimFormat)VTF[r][c]).getActualValue() ==
								((ValueTrimFormat.NumberTrimFormat)VTF[c][r]).getActualValue() * (-1);
						else
							skewSymmetric = VTF[r][c].isNotSet();
					}
				}
			}

			if(symmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				transferSampleTriangular(isUpperTriangular);
				isMapped = findMapping();

				// Upper Triangular
				if(!isMapped) {
					isUpperTriangular = true;
					retrieveSample();
					//sampleMatrix = sampleMatrixClone;
					transferSampleTriangular(isUpperTriangular);
					isMapped = findMapping();
				}
			}
			// Skew-Symmetric check:
			else if(skewSymmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				skewCoefficient = 1;
				transferSampleTriangular(isUpperTriangular);
				isMapped = findMapping();

				// Lower Triangular Skew
				if(!isMapped) {
					skewCoefficient = -1;
					transferSampleSkew(skewCoefficient);
					VTF = convertSampleTOValueTrimFormat();
					isMapped = findMapping();
				}

				// Upper Triangular
				if(!isMapped) {
					isUpperTriangular = true;
					skewCoefficient = 1;
					retrieveSample();
					transferSampleTriangular(isUpperTriangular);
					VTF = convertSampleTOValueTrimFormat();
					isMapped = findMapping();
				}
				// Upper Triangular Skew
				if(!isMapped) {
					skewCoefficient = -1;
					transferSampleSkew(skewCoefficient);
					VTF = convertSampleTOValueTrimFormat();
					isMapped = findMapping();
				}
			}
		}
		mapped = isMapped;
	}

	protected boolean findMapping() {

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
			ValueTrimFormat[] vtfRow = new ValueTrimFormat[ncols];
			for(int i = 0; i < ncols; i++) {
				vtfRow[i] = VTF[r][i].getACopy();
			}
			Arrays.sort(vtfRow);

			for(ValueTrimFormat vtf : vtfRow) {

				if(vtf.isNotSet()) {
					continue;
				}

				int c = vtf.getColIndex();
				HashSet<Integer> checkedLines = new HashSet<>();
				while(checkedLines.size() < nlines) {
					String row = sampleRawRows.get(itRow);
					NumberMappingInfo nmi = getCellMapping(row, vtf, colIndexes, colIndexSizes);
					if(nmi.mapped) {
						mapRow[r][c] = itRow;
						mapCol[r][c] = nmi.index;
						mapSize[r][c] = nmi.size;
						colIndexes.add(nmi.index);
						colIndexSizes.add(nmi.size);
						break;
					}
					else {
						checkedLines.add(itRow);
						itRow++;
						if(itRow == nlines)
							itRow = 0;
						colIndexes.clear();
						colIndexSizes.clear();

						for(int i = 0; i < nrows; i++) {
							for(int j = 0; j < ncols; j++) {
								if(mapRow[i][j] == itRow) {
									colIndexes.add(mapCol[i][j]);
									colIndexSizes.add(mapSize[i][j]);
								}
							}
						}
					}
				}
			}
		}
		boolean flagMap = true;
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] == -1 && !VTF[r][c].isNotSet())
					flagMap = false;
		return flagMap;
	}

	// Looking for a value(ValueTrimFormat) on a String
	public NumberMappingInfo getCellMapping(String row, ValueTrimFormat vtf, ArrayList<Integer> reservedIndex,
		ArrayList<Integer> reservedSize) {

		NumberMappingInfo result = null;
		ArrayList<String> rowChunks = new ArrayList<>();
		ArrayList<Integer> baseIndexes = new ArrayList<>();

		BitSet bitSet = new BitSet(row.length());
		for(int i = 0; i < reservedIndex.size(); i++) {
			bitSet.set(reservedIndex.get(i), reservedIndex.get(i) + reservedSize.get(i));
		}
		getChunksOFString(row, bitSet, rowChunks, baseIndexes);

		int rci = -1;
		for(String rc : rowChunks) {
			rci++;
			result = vtf.getMappingInfo(rc);
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

	public final CustomProperties getFormatProperties() throws Exception {
		CustomProperties ffp = null;
		if(isRowRegular()) {
			ffp = getFileFormatPropertiesOfRRCRMapping();
			if(ffp == null) {
				ffp = getFileFormatPropertiesOfRRCIMapping();
			}
		}
		else {
			ffp = getFileFormatPropertiesOfRIMapping();
		}

		return ffp;
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
	public final CustomProperties getFileFormatPropertiesOfRRCRMapping() {

		int nrows = mapRow.length;
		int ncols = mapRow[0].length;

		String[][] delims = new String[nrows][ncols + 1];
		for(int r = 0; r < nrows; r++) {
			int[] rCol = new int[ncols];
			System.arraycopy(mapCol[r], 0, rCol, 0, ncols);
			String row = sampleRawRows.get(r);

			HashMap<Integer, Integer> ciMap = new HashMap<>();
			HashMap<Integer, Integer> icMap = new HashMap<>();
			HashMap<Integer, String> idMap = new HashMap<>();
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

				if(sIndex != 0 && subRow.length() >= 1) {
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

		FastStringTokenizer fastStringTokenizer;
		while(token.length() < maxSizeOfToken) {
			token.append(stringToken.charAt(token.length()));

			boolean flagCurrToken = true;
			HashSet<String> ns = new HashSet<>();
			for(int r = 0; r < nrows; r++) {
				String row = rowDelims.get(r);
				fastStringTokenizer = new FastStringTokenizer(token.toString());
				fastStringTokenizer.reset(row);
				ArrayList<String> delimsOfToken = fastStringTokenizer.getTokens();
				ns.addAll(delimsOfToken);
				if(fastStringTokenizer._count != ncols - 1) {
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
			CustomProperties ffpgr = new CustomProperties(CustomProperties.GRPattern.Regular, uniqueDelimiter,
				naString);
			ffpgr.setDescription("CSV Format Recognized");
			ffpgr.setFirstIndex(0);
			return ffpgr;
		}
		else
			return null;
	}

	private static class FastStringTokenizer implements Serializable {
		private static final long serialVersionUID = -4698672725609750097L;
		private String _string = null;
		private String _del = "";
		private int _pos = -1;
		private int _count = 0;

		public FastStringTokenizer(String delimiter) {
			_del = delimiter;
			reset(null);
		}

		public void reset(String string) {
			_string = string;
			_pos = 0;
			_count = 0;
		}

		private String nextToken() {
			int len = _string.length();
			int start = _pos;

			//find start (skip over leading delimiters)
			while(start != -1 && start < len && _del.equals(_string.substring(start, start + _del.length()))) {
				start += _del.length();
				_count++;
			}

			//find end (next delimiter) and return
			if(start < len && start != -1) {
				_pos = _string.indexOf(_del, start);
				if(start < _pos && _pos < len) {
					return _string.substring(start, _pos);
				}
				else
					return _string.substring(start);
			}
			//no next token
			return null;
		}

		public ArrayList<String> getTokens() {
			ArrayList<String> tokens = new ArrayList<>();
			tokens.add("");
			String token;
			do {
				token = nextToken();
				if(token != null) {
					tokens.add(token);
				}
			}
			while(token != null);
			return tokens;
		}
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
				}
			}
			else
				break;
		}
		return mergeCount;
	}

	private CustomProperties getFileFormatPropertiesOfRIMapping() throws Exception {

		int firstRowIndex = 0;
		int firstColIndex = 0;

		CustomProperties ffp = getDelimsOfMapping(firstRowIndex, firstColIndex);

		// FirstRowIndex = 1, FirstColIndex = 1
		if(ffp == null) {
			firstRowIndex = 1;
			firstColIndex = 1;
			ffp = getDelimsOfMapping(firstRowIndex, firstColIndex);
		}

		// FirstRowIndex = 1, FirstColIndex = 0
		if(ffp == null) {
			firstRowIndex = 1;
			firstColIndex = 0;
			ffp = getDelimsOfMapping(firstRowIndex, firstColIndex);
		}

		// FirstRowIndex = 0, FirstColIndex = 1
		if(ffp == null) {
			firstRowIndex = 0;
			firstColIndex = 1;
			ffp = getDelimsOfMapping(firstRowIndex, firstColIndex);
		}

		if(ffp != null) {
			ffp.setDescription(
				"Market Matrix Format Recognized: FirstRowIndex: " + firstRowIndex + " and  FirstColIndex: " + firstColIndex);
			ffp.setFirstIndex(firstRowIndex);
		}

		return ffp;
	}

	private CustomProperties getDelimsOfMapping(int firstRowIndex, int firstColIndex) throws Exception {

		HashSet<Integer> checkedRow = new HashSet<>();
		RowColValue[] mapRCV = new RowColValue[(int) nnz];

		boolean rcvMapped = true;
		int nzIndex = 0;
		for(int r = nrows - 1; r >= 0 && rcvMapped; r--) {
			for(int c = ncols - 1; c >= 0 && rcvMapped; c--) {
				if(VTF[r][c].isNotSet())
					continue;
				RowColValue rcv = new RowColValue(r + firstRowIndex, c + firstColIndex, VTF[r][c]);
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
				while(++index < nlines && !rcv.isMapped());
				rcvMapped = rcv.isMapped();
			}
		}

		if(!rcvMapped) {
			return null;
		}
		else {
			HashSet<String> delims = new HashSet<>();
			mapRCV[0].findDelims();
			delims.add(mapRCV[0].colDelim);
			delims.add(mapRCV[0].valueDelim);
			int minDelimLength = Math.min(mapRCV[0].colDelim.length(), mapRCV[0].valueDelim.length());
			for(int i = 1; i < nnz; i++) {
				mapRCV[i].findDelims();
				delims.add(mapRCV[i].colDelim);
				delims.add(mapRCV[i].valueDelim);
			}

			String uniqueDelim = null;

			for(int l = 1; l < minDelimLength + 1; l++) {
				boolean flagToken = true;
				HashSet<String> token = new HashSet<>();
				Iterator<String> it = delims.iterator();
				while(it.hasNext()) {
					String delim = it.next();
					if(delim.length() % l != 0) {
						flagToken = false;
						break;
					}
					for(int i = 0; i <= delim.length() - l; i += l)
						token.add(delim.substring(i, i + l));
					if(token.size() > 1) {
						flagToken = false;
						break;
					}
				}
				if(flagToken) {
					if(token.size() > 0)
						uniqueDelim = token.iterator().next();
					break;
				}
			}

			if(uniqueDelim != null) {
				CustomProperties.GRSymmetry symmetry;
				if(symmetric)
					symmetry = CustomProperties.GRSymmetry.SYMMETRIC;
				else if(skewSymmetric)
					symmetry = CustomProperties.GRSymmetry.SKEW_SYMMETRIC;
				else
					symmetry = CustomProperties.GRSymmetry.GENERAL;

				return new CustomProperties(symmetry, uniqueDelim);
			}
			else
				return null;
		}
	}

	public CustomProperties getFileFormatPropertiesOfRRCIMapping() {

		CustomProperties ffplibsvm;
		int firstColIndex = 0;

		//FirstColIndex = 0
		ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);

		//FirstColIndex = 1
		if(ffplibsvm == null) {
			firstColIndex = 1;
			ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);
		}

		if(ffplibsvm != null) {
			ffplibsvm.setDescription("LibSVM Format Recognized: First Index Started From " + firstColIndex);
			ffplibsvm.setFirstIndex(firstColIndex);
		}
		return ffplibsvm;
	}

	private CustomProperties getDelimsOfRRCIMapping(int firstColIndex) {
		HashMap<String, HashSet<String>> tokens = new HashMap<>();
		HashSet<String> allTokens = new HashSet<>();

		for(int c = ncols - 1; c >= 0; c--) {
			ValueTrimFormat v = VTF[0][c];
			if(v.isNotSet())
				continue;

			String key = (c + firstColIndex) + "," + v.getStringOfActualValue();
			HashSet<String> token = tokens.get(key);

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
				separator = verifyRowWithIndexDelim(r, indexSeparator, labelIndex.get(i), firstColIndex);
				if(separator == null) {
					isVerify = false;
					break;
				}
			}
		}
		if(isVerify) {
			return new CustomProperties(CustomProperties.GRPattern.Regular, separator, indexSeparator);
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
				ValueTrimFormat.NumberTrimFormat vtfColIndex = new ValueTrimFormat.NumberTrimFormat(c + firstColIndex);
				ValueTrimFormat vtfColValue = VTF[rowID][c];

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
							nmiIndex = vtfColIndex.isNotSet() ? vtfColIndex
								.getMappingInfoIncludeZero(subChunk) : vtfColIndex.getMappingInfo(subChunk);

							if(!nmiIndex.mapped)
								break;

							nmiIndex.size += mergeDelimiters(vtfColIndex.getActualValue(),
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
						NumberMappingInfo miValue = vtfColValue.getMappingInfo(subChunk);
						if(vtfColValue instanceof ValueTrimFormat.NumberTrimFormat) {
							miValue.size += mergeDelimiters(
								((ValueTrimFormat.NumberTrimFormat) vtfColValue).getActualValue(),
								subChunk.substring(miValue.index, miValue.index + miValue.size),
								subChunk.substring(miValue.index + miValue.size));
						}

						if(miValue.mapped) {
							itemVerify = true;
							nmiIndex.index += sPosition + baseIndexes.get(i);
							if(labelIndex - firstColIndex != c) {
								miValue.index += indexDelim.length();
							}
							miValue.index += nmiIndex.index + nmiIndex.size;
							bitSet.set(nmiIndex.index, miValue.index + miValue.size);
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
		HashSet<String> separators = new HashSet<>();
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
		for(int i = 0; i < length; ) {
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
		private final ValueTrimFormat.NumberTrimFormat vtfRow;
		private final ValueTrimFormat.NumberTrimFormat vtfCol;
		private final ValueTrimFormat vtfValue;
		private NumberMappingInfo miRow;
		private NumberMappingInfo miCol;
		private NumberMappingInfo miValue;
		private boolean mapped;
		private String rowString;

		private String rowDelim;
		private String colDelim;
		private String valueDelim;

		public RowColValue(int row, int col, ValueTrimFormat value) {
			rowString = null;
			// Row, Col, Value
			vtfRow = new ValueTrimFormat.NumberTrimFormat(row);
			vtfCol = new ValueTrimFormat.NumberTrimFormat(col);
			vtfValue = value.getACopy();
			mapped = false;
		}

		/* valid formats:
		   Row, Col, Value
		1. 0  , 0  , Value  >> 110 -> 6
		2. 0  , col, Value  >> 100 -> 4
		3. row, 0  , value  >> 010 -> 2
		4. row, col, value  >> 000 -> 0
		-----------------   >> otherwise the value is not set.
		 */
		public boolean isMapped(String row) {

			byte hasZero = 0b000;
			if(vtfRow.isNotSet())
				hasZero |= 0b100;

			if(vtfCol.isNotSet())
				hasZero |= 0b010;

			if(vtfValue.isNotSet())
				hasZero |= 0b001;

			switch(hasZero) {
				case 0:
					miRow = vtfRow.getMappingInfo(row);
					if(!miRow.mapped)
						return false;

					miCol = vtfCol.getMappingInfo(row.substring(miRow.index + miRow.size));
					if(!miCol.mapped)
						return false;
					miCol.index += miRow.index + miRow.size;

					miValue = vtfValue.getMappingInfo(row.substring(miCol.index + miCol.size));
					if(!miValue.mapped)
						return false;
					miValue.index += miCol.index + miCol.size;
					break;

				case 2:
					miRow = vtfRow.getMappingInfo(row);
					if(!miRow.mapped)
						return false;

					miValue = vtfValue.getMappingInfo(row.substring(miRow.index + miRow.size));
					if(!miValue.mapped)
						return false;
					miValue.index += miRow.index + miRow.size;

					miCol = vtfCol.getMappingInfoIncludeZero(row.substring(miRow.index + miRow.size, miValue.index));
					if(!miCol.mapped)
						return false;
					miCol.index += miRow.index + miRow.size;
					break;

				case 4:
					miCol = vtfCol.getMappingInfo(row);
					if(!miCol.mapped)
						return false;

					miValue = vtfValue.getMappingInfo(row.substring(miCol.index + miCol.size));
					if(!miValue.mapped)
						return false;
					miValue.index += miCol.index + miCol.size;

					miRow = vtfRow.getMappingInfoIncludeZero(row.substring(0, miCol.index));
					if(!miRow.mapped)
						return false;
					break;

				case 6:
					miValue = vtfValue.getMappingInfo(row);
					if(!miValue.mapped)
						return false;

					miRow = vtfRow.getMappingInfoIncludeZero(row.substring(0, miValue.index));
					if(!miRow.mapped)
						return false;

					miCol = vtfCol.getMappingInfoIncludeZero(row.substring(miRow.index + miRow.size));
					if(!miCol.mapped)
						return false;
					miCol.index += miRow.index + miRow.size;
					break;

				default:
					throw new RuntimeException("Not set values can't be find on a string");
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

			rowDelim = rowString.substring(0, miRow.index);
			miRow.size += mergeDelimiters(vtfRow.getActualValue(),
				rowString.substring(miRow.index, miRow.index + miRow.size),
				rowString.substring(miRow.index + miRow.size));

			colDelim = rowString.substring(miRow.index + miRow.size, miCol.index);
			miCol.size += mergeDelimiters(vtfCol.getActualValue(),
				rowString.substring(miCol.index, miCol.index + miCol.size),
				rowString.substring(miCol.index + miCol.size));

			valueDelim = rowString.substring(miCol.index + miCol.size, miValue.index);

			if(vtfValue instanceof ValueTrimFormat.NumberTrimFormat) {
				miValue.size += mergeDelimiters(((ValueTrimFormat.NumberTrimFormat) vtfValue).getActualValue(),
					rowString.substring(miValue.index, miValue.index + miValue.size),
					rowString.substring(miValue.index + miValue.size));
			}

		}

		public String getRowDelim() {
			return rowDelim;
		}

		public String getColDelim() {
			return colDelim;
		}

		public String getValueDelim() {
			return valueDelim;
		}

		public ValueTrimFormat.NumberTrimFormat getVtfRow() {
			return vtfRow;
		}

		public ValueTrimFormat.NumberTrimFormat getVtfCol() {
			return vtfCol;
		}

		public ValueTrimFormat getVtfValue() {
			return vtfValue;
		}
	}

	class ColIndexValue {
		private final ValueTrimFormat.NumberTrimFormat vtfColIndex;
		private final ValueTrimFormat vtfColValue;
		private NumberMappingInfo miColIndex;
		private NumberMappingInfo miColValue;
		private boolean mapped;
		private String indSep;
		private String separator;

		public ColIndexValue(int col, ValueTrimFormat value) {
			vtfColIndex = new ValueTrimFormat.NumberTrimFormat(col);
			vtfColValue = value.getACopy();
			indSep = null;
			separator = null;
			mapped = false;
			miColIndex = null;
			miColValue = null;
		}

		public ColIndexValue(ValueTrimFormat.NumberTrimFormat vtfColIndex, ValueTrimFormat vtfColValue,
			NumberMappingInfo miColIndex, NumberMappingInfo miColValue, String indSep) {
			this.vtfColIndex = vtfColIndex;
			this.vtfColValue = vtfColValue;
			this.miColIndex = miColIndex;
			this.miColValue = miColValue;
			this.indSep = indSep;
		}

		public HashSet<String> getMappedTokens(String row) {

			int sPosition = 0;
			HashSet<String> tokens = new HashSet<>();

			while(sPosition < row.length()) {
				NumberMappingInfo nmiIndex;

				nmiIndex = vtfColIndex.isNotSet() ? vtfColIndex
					.getMappingInfoIncludeZero(row.substring(sPosition)) : vtfColIndex
					.getMappingInfo(row.substring(sPosition));
				if(!nmiIndex.mapped) {
					break;
				}
				nmiIndex.index += sPosition;

				nmiIndex.size += mergeDelimiters(vtfColIndex.getActualValue(),
					row.substring(+nmiIndex.index, nmiIndex.index + nmiIndex.size),
					row.substring(nmiIndex.index + nmiIndex.size));

				NumberMappingInfo nmiValue = vtfColValue.getMappingInfo(row.substring(nmiIndex.index + nmiIndex.size));
				if(!nmiValue.mapped) {
					break;
				}
				nmiValue.index += nmiIndex.index + nmiIndex.size;

				if(vtfColValue instanceof ValueTrimFormat.NumberTrimFormat) {
					nmiValue.size += mergeDelimiters(((ValueTrimFormat.NumberTrimFormat) vtfColValue).getActualValue(),
						row.substring(nmiValue.index, nmiValue.index + nmiValue.size),
						row.substring(nmiValue.index + nmiValue.size));
				}

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
