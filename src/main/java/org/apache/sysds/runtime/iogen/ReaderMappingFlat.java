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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public abstract class ReaderMappingFlat extends ReaderMapping {

	protected int[][] mapRow;
	protected int[][] mapCol;
	protected boolean symmetric;
	protected boolean skewSymmetric;
	protected boolean isUpperTriangular;
	protected int skewCoefficient;
	protected final ArrayList<RawRow> sampleRawRows;

	protected final int nlines;
	protected int firstRowIndex;
	protected int firstColIndex;

	protected ValueTrimFormat[][] VTF;
	protected ValueTrimFormat[][] VTFClone = null;

	public ReaderMappingFlat(String raw) throws Exception {
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		int nlines = 0;
		sampleRawRows = new ArrayList<>();
		while((value = br.readLine()) != null) {
			sampleRawRows.add(new RawRow(value));
			nlines++;
		}
		this.nlines = nlines;
		firstColIndex = 0;
		firstRowIndex = 0;
	}

	protected abstract boolean isSchemaNumeric();

	protected void cloneSample() {
		if(VTFClone == null) {
			VTFClone = new ValueTrimFormat[nrows][ncols];
			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++)
					VTFClone[r][c] = VTF[r][c].getACopy();
		}
	}

	protected void retrieveSample() {
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				VTF[r][c] = VTFClone[r][c].getACopy();
	}

	protected void transferSampleTriangular(boolean isUpper) throws Exception {
		if(nrows != ncols)
			throw new Exception("For upper triangular both Row and Col should be same!");

		for(int r = 0; r < nrows; r++) {
			if(isUpper) {
				for(int c = 0; c < r; c++) {
					VTF[r][c].setNoSet();
				}
			}
			else {
				for(int c = r + 1; c < ncols; c++) {
					VTF[r][c].setNoSet();
				}
			}
		}
	}

	protected void transferSampleSkew(int coefficient) throws Exception {
		if(coefficient != 1 && coefficient != -1)
			throw new Exception("The value of Coefficient have to be 1 or -1!");

		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++) {
				if(!VTF[r][c].isNotSet() && VTF[r][c].getValueType().isNumeric())
					VTF[r][c] = new ValueTrimFormat(VTF[r][c].getColIndex(), VTF[r][c].getValueType(),
						VTF[r][c].getDoubleActualValue() * coefficient);
			}
	}

	protected abstract ValueTrimFormat[][] convertSampleTOValueTrimFormat();

	// Matrix Reader Mapping
	public static class MatrixReaderMapping extends ReaderMappingFlat {

		private MatrixBlock sampleMatrix;

		public MatrixReaderMapping(String raw, MatrixBlock matrix) throws Exception {
			super(raw);
			this.sampleMatrix = matrix;
			nrows = sampleMatrix.getNumRows();
			ncols = sampleMatrix.getNumColumns();
			VTF = convertSampleTOValueTrimFormat();
			runMapping();
		}

		// Convert: convert each value of a sample matrix to NumberTrimFormat
		@Override
		protected ValueTrimFormat[][] convertSampleTOValueTrimFormat() {
			ValueTrimFormat[][] result = new ValueTrimFormat[nrows][ncols];
			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++) {
					result[r][c] = new ValueTrimFormat(c, Types.ValueType.FP64, sampleMatrix.getValue(r, c));
				}
			return result;
		}

		@Override
		protected boolean isSchemaNumeric() {
			return true;
		}

	}

	// Frame Reader Mapping
	public static class FrameReaderMapping extends ReaderMappingFlat {

		private FrameBlock sampleFrame;
		private Types.ValueType[] schema;

		public FrameReaderMapping(String raw, FrameBlock frame) throws Exception {
			super(raw);
			this.sampleFrame = frame;
			nrows = sampleFrame.getNumRows();
			ncols = sampleFrame.getNumColumns();
			schema = sampleFrame.getSchema();
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
					result[r][c] = new ValueTrimFormat(c, schema[c], sampleFrame.get(r, c));
				}
			return result;
		}

		@Override
		protected boolean isSchemaNumeric() {
			boolean result = true;
			for(Types.ValueType vt : schema)
				result &= vt.isNumeric();
			return result;
		}
	}

	public void runMapping() throws Exception {

		mapped = findMapping();
		boolean schemaNumeric = isSchemaNumeric();
		if(!mapped) {
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
							skewSymmetric = VTF[r][c].getDoubleActualValue() == VTF[c][r].getDoubleActualValue() * -1;
						else
							skewSymmetric = VTF[r][c].isNotSet();
					}
				}
			}

			boolean isRR = isRowRegular();
			if(symmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				transferSampleTriangular(isUpperTriangular);
				mapped = isRR ? findMapping() : findMapping() && verifyRISymmetricMapping(isUpperTriangular);

				// Upper Triangular
				if(!mapped) {
					isUpperTriangular = true;
					retrieveSample();
					transferSampleTriangular(isUpperTriangular);
					mapped = isRR ? findMapping() : findMapping() && verifyRISymmetricMapping(isUpperTriangular);
				}
			}
			// Skew-Symmetric check:
			else if(skewSymmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				transferSampleTriangular(isUpperTriangular);
				mapped = isRR ? findMapping() : findMapping() && verifyRISymmetricMapping(isUpperTriangular);

				// Lower Triangular Skew
				if(!mapped) {
					skewCoefficient = -1;
					transferSampleSkew(skewCoefficient);
					mapped = isRR ? findMapping() : findMapping() && verifyRISymmetricMapping(isUpperTriangular);
				}

				// Upper Triangular
				if(!mapped) {
					isUpperTriangular = true;
					skewCoefficient = 1;
					retrieveSample();
					transferSampleTriangular(isUpperTriangular);
					mapped = isRR ? findMapping() : findMapping() && verifyRISymmetricMapping(isUpperTriangular);
				}
				// Upper Triangular Skew
				if(!mapped) {
					skewCoefficient = -1;
					transferSampleSkew(skewCoefficient);
					mapped = isRR ? findMapping() : findMapping() && verifyRISymmetricMapping(isUpperTriangular);
				}
			}
		}
	}

	protected boolean findMapping() {
		mapRow = new int[nrows][ncols];
		mapCol = new int[nrows][ncols];

		// Set "-1" as default value for all defined matrix
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				mapRow[r][c] = mapCol[r][c] = -1;

		for(int i = 0; i < nlines; i++) {
			sampleRawRows.get(i).resetReserved();
		}
		int itRow = 0;
		for(int r = 0; r < nrows; r++) {
			ArrayList<ValueTrimFormat> vtfRow = new ArrayList<>();
			for(int i = 0; i < ncols; i++) {
				if(!VTF[r][i].isNotSet())
					vtfRow.add(VTF[r][i]);
			}
			Collections.sort(vtfRow);

			for(ValueTrimFormat vtf : vtfRow) {
				int c = vtf.getColIndex();
				HashSet<Integer> checkedLines = new HashSet<>();
				while(checkedLines.size() < nlines) {
					RawRow row = sampleRawRows.get(itRow);
					Pair<Integer, Integer> mi = row.findValue(vtf, false);
					if(mi.getKey() != -1) {
						mapRow[r][c] = itRow;
						mapCol[r][c] = mi.getKey();
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
		boolean flagMap = true;
		for(int r = 0; r < nrows && flagMap; r++)
			for(int c = 0; c < ncols && flagMap; c++)
				if(mapRow[r][c] == -1 && !VTF[r][c].isNotSet()) {
					flagMap = false;
				}
		return flagMap;
	}

	private boolean verifyRISymmetricMapping(boolean upperTriangular) {

		boolean result = false;
		int[] rowIndex = {0, 1, 0, 1};
		int[] colIndex = {0, 1, 1, 0};
		for(int i = 0; i < rowIndex.length && !result; i++) {
			result = verifyRISymmetricMapping(upperTriangular, rowIndex[i], colIndex[i]);
			if(result) {
				firstRowIndex = rowIndex[i];
				firstColIndex = colIndex[i];
			}
		}
		return result;
	}

	private boolean verifyRISymmetricMapping(boolean upperTriangular, int firstRowIndex, int firstColIndex) {

		HashSet<Integer> checkedRow = new HashSet<>();
		boolean rcvMapped = true;
		int selectedIndex;

		for(int r = nrows - 2; r >= 0 && rcvMapped; r--) {
			selectedIndex = upperTriangular ? Math.min(r + 1, nrows - 1) : Math.max(r - 1, 0);
			if(r == selectedIndex)
				break;
			int lindeIndex = 0;
			rcvMapped = false;
			do {
				if(checkedRow.contains(lindeIndex) || VTF[r][selectedIndex].isNotSet())
					continue;
				RawRow row = sampleRawRows.get(lindeIndex).getResetClone();
				if(isMapRowColValue(row, r + firstRowIndex, selectedIndex + firstColIndex, VTF[r][selectedIndex])) {
					checkedRow.add(lindeIndex);
					rcvMapped = true;
				}
			}
			while(++lindeIndex < nlines && !rcvMapped);
		}
		return rcvMapped;
	}

	@Override
	public CustomProperties getFormatProperties() throws Exception {
		CustomProperties ffp;
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

		for(int r = 0; r < nrows && result; r++) {
			for(int c = 0; c < ncols && result; c++) {
				if(mapRow[r][c] != -1 && mapRow[r][c] != rValue + r) {
					result = false;
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

		ArrayList<String> rowDelims = new ArrayList<>();
		HashSet<String> naString = new HashSet<>();
		String stringToken = null;

		// append all delimiters as a string and then tokenize it
		for(int r = 0; r < nrows; r++) {
			RawRow rr = sampleRawRows.get(r);
			Pair<String, String> pair = rr.getDelims();
			rowDelims.add(pair.getValue());
			if(stringToken == null || (pair.getKey().length() > 0 && stringToken.length() > pair.getKey().length()))
				stringToken = pair.getKey();
		}
		if(stringToken.length() == 0)
			stringToken = rowDelims.get(0);
		String uniqueDelimiter = null;
		StringBuilder token = new StringBuilder();

		FastStringTokenizer fastStringTokenizer;

		for(Character ch : stringToken.toCharArray()) {
			token.append(ch);
			boolean flagCurrToken = true;
			HashSet<String> ns = new HashSet<>();
			fastStringTokenizer = new FastStringTokenizer(token.toString());
			for(int r = 0; r < nrows; r++) {
				String row = rowDelims.get(r);
				fastStringTokenizer.reset(row);
				ArrayList<String> delimsOfToken = fastStringTokenizer.getTokens();

				// remove numeric NA Strings
				// This case can appear in Frame DataType
				for(String s : delimsOfToken) {
					try {
						Double.parseDouble(s);
					}
					catch(Exception ex) {
						ns.add(s);
					}
				}
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
			while(start != -1 && start < len && _del
				.equals(_string.substring(start, Math.min(start + _del.length(), _string.length())))) {
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

	private CustomProperties getFileFormatPropertiesOfRIMapping() {

		int[] rowIndex = {0, 1, 0, 1};
		int[] colIndex = {0, 1, 1, 0};
		CustomProperties ffp = null;
		for(int i = 0; i < rowIndex.length && ffp == null; i++) {
			ffp = getDelimsOfMapping(rowIndex[i], colIndex[i]);
			if(ffp != null) {
				firstRowIndex = rowIndex[i];
				firstColIndex = colIndex[i];
			}
		}

		if(ffp != null) {
			ffp.setFirstColIndex(firstColIndex);
			ffp.setFirstRowIndex(firstRowIndex);
			ffp.setDescription(
				"Market Matrix Format Recognized: FirstRowIndex: " + firstRowIndex + " and  FirstColIndex: " + firstColIndex);
		}
		return ffp;
	}

	private CustomProperties getDelimsOfMapping(int firstRowIndex, int firstColIndex) {

		//HashSet<Integer> checkedRow = new HashSet<>();
		HashSet<String> delims = new HashSet<>();
		int minDelimLength = -1;
		boolean rcvMapped = false;
		int selectedRowIndex = nrows - 2;
		int selectedColIndex = ncols - 1;
		// select maximum none zero col index
		for(int c = ncols - 1; c >= 0; c--) {
			if(!VTF[selectedRowIndex][c].isNotSet()) {
				selectedColIndex = c;
				break;
			}
		}
		int lindeIndex = 0;
		do {
			RawRow row = sampleRawRows.get(lindeIndex).getResetClone();
			if(isMapRowColValue(row, selectedRowIndex + firstRowIndex, selectedColIndex + firstColIndex,
				VTF[selectedRowIndex][selectedColIndex])) {
				rcvMapped = true;

				Pair<HashSet<String>, Integer> pair = row.getDelimsSet();
				delims.addAll(pair.getKey());
				minDelimLength = minDelimLength == -1 ? pair.getValue() : Math.min(minDelimLength, pair.getValue());
			}
		}
		while(++lindeIndex < nlines && !rcvMapped);

		if(!rcvMapped) {
			return null;
		}
		else {

			String uniqueDelim = null;
			for(int l = 1; l < minDelimLength + 1; l++) {
				boolean flagToken = true;
				HashSet<String> token = new HashSet<>();
				for(String delim : delims) {
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

				return new CustomProperties(symmetry, uniqueDelim, firstRowIndex, firstColIndex);
			}
			else
				return null;
		}
	}

	public CustomProperties getFileFormatPropertiesOfRRCIMapping() {

		CustomProperties ffplibsvm;
		int firstColIndex = 0;

		// FirstColIndex = 0
		ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);

		// FirstColIndex = 1
		if(ffplibsvm == null) {
			firstColIndex = 1;
			ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);
		}

		if(ffplibsvm != null) {
			ffplibsvm.setDescription("LibSVM Format Recognized: First Index Started From " + firstColIndex);
			ffplibsvm.setFirstColIndex(firstColIndex);
		}
		return ffplibsvm;
	}

	private CustomProperties getDelimsOfRRCIMapping(int firstColIndex) {
		HashMap<String, HashSet<String>> tokens = new HashMap<>();
		HashSet<String> allTokens = new HashSet<>();
		int maxNNZCount = 0;
		int selectedRowIndex = 0;
		for(int r = 0; r < nrows; r++) {
			int rnnz = 0;
			for(int c = 0; c < ncols; c++)
				if(!VTF[r][c].isNotSet())
					rnnz++;
			if(maxNNZCount < rnnz) {
				maxNNZCount = rnnz;
				selectedRowIndex = r;
			}
		}

		RawRow row = sampleRawRows.get(selectedRowIndex);
		// For find index delimiter, we need to find all possible "Index Delim Value" tokens
		for(int c = ncols - 1; c >= 0; c--) {
			ValueTrimFormat v = VTF[selectedRowIndex][c];
			if(v.isNotSet())
				continue;

			String key = (c + firstColIndex) + "," + v.getStringOfActualValue();
			HashSet<String> token = tokens.computeIfAbsent(key, k -> new HashSet<>());
			token.addAll(getColIndexValueMappedTokens(row, c + firstColIndex, v));
			allTokens.addAll(token);
		}

		//After find all tokens the intersection of tokens is a good candidate for "Index delimiter"
		// This part of code try to find the intersection of tokens
		// In some cases like LobSVM label value don't have Index Delim token,
		// So, we ignored this condition for some values
		ArrayList<String> missedKeys = new ArrayList<>();
		HashSet<Integer> labelIndex = new HashSet<>();
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

		/* After find index delim token, the next step is find Item Separator
		 The algorithm for find separator, mark all Indexes, Values and Index Delim on the raw string
		 Finally the reminder of the text is separator. In some cases(i.e., duplicated values)
		 there are more than on position for value and this cause wrong matching and finally wrong value
		 for separator. To avoid this type of problems, first looked for biggest char base size values
		 (for example a= 123.45 b= 1000000 a will match first because based on VariableTrimFormat algorithm
		 "a" have 5 char ad the length is 5, but b have 1 char and the length is one).
		 */
		String separator = null;
		String indexSeparator = null;
		boolean isVerify = false;

		// Just one row of the sample raw is enough for finding item separator. "selectedRowIndex" mentioned
		// first row of sample raw data

		for(int i = 0; i < selectedTokens.size() && !isVerify; i++) {
			isVerify = true;
			indexSeparator = selectedTokens.get(i);

			row = sampleRawRows.get(selectedRowIndex).getResetClone();
			// find all values
			ArrayList<ValueTrimFormat> vtfValueList = new ArrayList<>();
			ValueTrimFormat vtfIndexDelim = new ValueTrimFormat(indexSeparator);
			for(int c = 0; c < ncols; c++) {
				if(!VTF[selectedRowIndex][c].isNotSet() && !labelIndex.contains(c + firstColIndex)) {
					vtfValueList.add(VTF[selectedRowIndex][c].getACopy());
				}
			}
			Collections.sort(vtfValueList);

			for(ValueTrimFormat vtf : vtfValueList) {
				ArrayList<ValueTrimFormat> indexDelimValue = new ArrayList<>();
				ValueTrimFormat vtfColIndex = new ValueTrimFormat(vtf.getColIndex() + firstColIndex);
				indexDelimValue.add(vtfColIndex);
				indexDelimValue.add(vtfIndexDelim);
				indexDelimValue.add(vtf);
				row.findSequenceValues(indexDelimValue, 0, true);
			}
			for(Integer li : labelIndex) {
				row.findValue(VTF[selectedRowIndex][li - firstColIndex], false);
			}
			separator = row.getDelims().getKey();
			if(separator == null) {
				isVerify = false;
				break;
			}
		}
		if(isVerify) {
			return new CustomProperties(CustomProperties.GRPattern.Regular, separator, indexSeparator, firstColIndex);
		}
		else
			return null;
	}

	private static boolean isMapRowColValue(RawRow rawrow, int row, int col, ValueTrimFormat value) {
		ValueTrimFormat vtfRow = new ValueTrimFormat(row);
		ValueTrimFormat vtfCol = new ValueTrimFormat(col);
		ValueTrimFormat vtfValue = value.getACopy();
		boolean mapped = true;

		byte hasZero = 0b000;
		if(vtfRow.isNotSet())
			hasZero |= 0b100;

		if(vtfCol.isNotSet())
			hasZero |= 0b010;

		if(vtfValue.isNotSet())
			hasZero |= 0b001;

		ValueTrimFormat[] order = new ValueTrimFormat[3];
		/* valid formats:
		   Row, Col, Value
		1. 0  , 0  , Value  >> 110 -> 6
		2. 0  , col, Value  >> 100 -> 4
		3. row, 0  , value  >> 010 -> 2
		4. row, col, value  >> 000 -> 0
		-----------------   >> otherwise the value is not set.
		 */
		switch(hasZero) {
			case 0:
				order[0] = vtfRow;
				order[1] = vtfCol;
				order[2] = vtfValue;
				break;

			case 2:
				order[0] = vtfRow;
				order[1] = vtfValue;
				order[2] = vtfCol;
				break;

			case 4:
				order[0] = vtfCol;
				order[1] = vtfValue;
				order[2] = vtfRow;
				break;

			case 6:
				order[0] = vtfValue;
				order[1] = vtfRow;
				order[2] = vtfCol;
				break;
			default:
				throw new RuntimeException("Not set values can't be find on a string");
		}
		for(ValueTrimFormat vtf : order) {
			if(rawrow.findValue(vtf, false).getKey() == -1) {
				mapped = false;
				break;
			}
		}
		return mapped;

	}

	private static HashSet<String> getColIndexValueMappedTokens(RawRow rawrow, int col, ValueTrimFormat value) {
		ValueTrimFormat vtfColIndex = new ValueTrimFormat(col);
		ValueTrimFormat vtfColValue = value.getACopy();
		Pair<Integer, Integer> pairCol;
		Pair<Integer, Integer> pairValue;
		HashSet<String> tokens = new HashSet<>();
		RawRow row = rawrow.getResetClone();
		int lastIndex = 0;
		int lastTokenStart = -1;
		int lastTokenEnd = -1;
		int lastTokenID = -1;
		do {
			row.resetReserved();
			row.setLastIndex(lastIndex);
			pairCol = row.findValue(vtfColIndex, true);
			if(pairCol.getKey() == -1)
				break;

			lastIndex = row.getNumericLastIndex();

			pairValue = row.findValue(vtfColValue, true);
			if(pairValue.getKey() == -1)
				break;

			int tl = pairValue.getKey() - pairCol.getKey() + pairCol.getValue();
			if(tl > 0) {

				if(lastTokenID == -1)
					lastTokenID = pairValue.getKey();

				if(lastTokenID != pairValue.getKey()) {
					String token = row.getRaw().substring(lastTokenStart, lastTokenEnd);
					tokens.add(token);
				}

				lastTokenStart = pairCol.getKey() + pairCol.getValue();
				lastTokenEnd = pairValue.getKey();
			}
		}
		while(true);
		if(lastTokenEnd - lastTokenStart > 0) {
			String token = row.getRaw().substring(lastTokenStart, lastTokenEnd);
			tokens.add(token);
		}
		return tokens;
	}

	public boolean isSymmetric() {
		return symmetric;
	}

}
