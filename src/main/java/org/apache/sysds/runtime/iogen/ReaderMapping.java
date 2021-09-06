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
import java.util.HashSet;
import java.util.HashMap;
import java.util.Arrays;
import java.util.Iterator;

public abstract class ReaderMapping {

	protected int[][] mapRow;
	protected int[][] mapCol;
	protected boolean symmetric;
	protected boolean skewSymmetric;
	protected boolean isUpperTriangular;
	protected int skewCoefficient;
	protected final ArrayList<RawRow> sampleRawRows;

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
			sampleRawRows.add(new RawRow(value));
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

		@Override
		protected void transferSampleTriangular(boolean isUpper) throws Exception {
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

		@Override
		protected void transferSampleSkew(int coefficient) throws Exception {
			if(coefficient != 1 && coefficient != -1)
				throw new Exception("The value of Coefficient have to be 1 or -1!");

			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++)
					sampleMatrix.setValue(r, c, sampleMatrix.getValue(r, c) * coefficient);
		}

		@Override
		protected boolean isSchemaNumeric() {
			return true;
		}

		@Override
		protected void cloneSample() {
			// Clone Sample Matrix
			sampleMatrixClone = new MatrixBlock(nrows, ncols, false);
			this.sampleMatrix.copy(0, nrows, 0, ncols, sampleMatrixClone, false);
		}

		@Override
		protected void retrieveSample() {
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

		@Override
		protected boolean isSchemaNumeric() {
			boolean result = true;
			for(Types.ValueType vt : schema)
				result &= vt.isNumeric();
			return result;
		}

		@Override
		protected void transferSampleTriangular(boolean isUpper) throws Exception {
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

		@Override
		protected void transferSampleSkew(int coefficient) throws Exception {
			if(coefficient != 1 && coefficient != -1)
				throw new Exception("The value of Coefficient have to be 1 or -1!");

			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++)
					sampleFrame.set(r, c, (Double) sampleFrame.get(r, c) * coefficient);
		}

		@Override
		protected void cloneSample() {
			// Clone SampleFrame
			sampleFrameClone = new FrameBlock(schema, names);
			sampleFrameClone.ensureAllocatedColumns(nrows);
			sampleFrame.copy(0, nrows, 0, ncols, sampleFrameClone);
		}

		@Override
		protected void retrieveSample() {
			// Retrieve SampleFrame from Cloned Data
			sampleFrameClone.copy(0, nrows, 0, ncols, sampleFrame);
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
							skewSymmetric = ((ValueTrimFormat.NumberTrimFormat) VTF[r][c])
								.getActualValue() == ((ValueTrimFormat.NumberTrimFormat) VTF[c][r])
								.getActualValue() * (-1);
						else
							skewSymmetric = VTF[r][c].isNotSet();
					}
				}
			}

			if(symmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				transferSampleTriangular(isUpperTriangular);
				mapped = findMapping();

				// Upper Triangular
				if(!mapped) {
					isUpperTriangular = true;
					retrieveSample();
					//sampleMatrix = sampleMatrixClone;
					transferSampleTriangular(isUpperTriangular);
					mapped = findMapping();
				}
			}
			// Skew-Symmetric check:
			else if(skewSymmetric) {
				// Lower Triangular
				isUpperTriangular = false;
				skewCoefficient = 1;
				transferSampleTriangular(isUpperTriangular);
				mapped = findMapping();

				// Lower Triangular Skew
				if(!mapped) {
					skewCoefficient = -1;
					transferSampleSkew(skewCoefficient);
					VTF = convertSampleTOValueTrimFormat();
					mapped = findMapping();
				}

				// Upper Triangular
				if(!mapped) {
					isUpperTriangular = true;
					skewCoefficient = 1;
					retrieveSample();
					transferSampleTriangular(isUpperTriangular);
					VTF = convertSampleTOValueTrimFormat();
					mapped = findMapping();
				}
				// Upper Triangular Skew
				if(!mapped) {
					skewCoefficient = -1;
					transferSampleSkew(skewCoefficient);
					VTF = convertSampleTOValueTrimFormat();
					mapped = findMapping();
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

		int itRow = 0;
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
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] == -1 && !VTF[r][c].isNotSet())
					flagMap = false;
		return flagMap;
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
			while(start != -1 && start < len &&
				_del.equals(_string.substring(start,Math.min(start + _del.length(), _string.length())))) {
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
		boolean rcvMapped = true;
		for(int r = nrows - 1; r >= 0 && rcvMapped; r--) {
			for(int c = ncols - 1; c >= 0 && rcvMapped; c--) {
				if(VTF[r][c].isNotSet())
					continue;
				int index = 0;
				rcvMapped = false;
				do {
					if(checkedRow.contains(index))
						continue;
					RawRow row = sampleRawRows.get(index).getResetClone();
					if(isMapRowColValue(row, r + firstRowIndex, c + firstColIndex, VTF[r][c])) {
						checkedRow.add(index);
						rcvMapped = true;
					}
				}
				while(++index < nlines && !rcvMapped);
				int a = 100;
			}
		}

		if(!rcvMapped) {
			return null;
		}
		else {
			HashSet<String> delims = new HashSet<>();
			int minDelimLength = -1;
			for(int i = 1; i < nnz; i++) {
				RawRow rr = sampleRawRows.get(i);
				if(rr.isMarked()) {
					Pair<HashSet<String>, Integer> pair = rr.getDelimsSet();
					delims.addAll(pair.getKey());
					minDelimLength = minDelimLength == -1 ? pair.getValue() : Math.min(minDelimLength, pair.getValue());
				}
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

		// FirstColIndex = 0
		ffplibsvm = getDelimsOfRRCIMapping(firstColIndex);

		// FirstColIndex = 1
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

		RawRow row = sampleRawRows.get(0).getResetClone();
		for(int c = ncols - 1; c >= 0; c--) {
			ValueTrimFormat v = VTF[0][c];
			if(v.isNotSet())
				continue;

			String key = (c + firstColIndex) + "," + v.getStringOfActualValue();
			HashSet<String> token = tokens.get(key);

			if(token == null) {
				token = new HashSet<>();
				tokens.put(key, token);
			}
			token.addAll(getColIndexValueMappedTokens(row, c + firstColIndex, v));
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

				row = sampleRawRows.get(r).getResetClone();
				// find all values
				for(int c = 0; c < ncols; c++) {
					if(VTF[r][c].isNotSet())
						continue;
					row.findValue(VTF[r][c], false);
				}
				for(int c = 0; c < ncols; c++) {
					if(VTF[r][c].isNotSet())
						continue;
					ValueTrimFormat.StringTrimFormat vtfIndexDelim = new ValueTrimFormat.StringTrimFormat(
						indexSeparator);
					row.findValue(vtfIndexDelim, false);
				}
				for(int c = 0; c < ncols; c++) {
					if(VTF[r][c].isNotSet())
						continue;
					ValueTrimFormat.NumberTrimFormat vtfColIndex = new ValueTrimFormat.NumberTrimFormat(
						c + firstColIndex);
					row.findValue(vtfColIndex, false);
				}
				separator = row.getDelims().getKey();
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

		//
		//		boolean isVerify = false;
		//		for(int i = 0; i < selectedTokens.size() && !isVerify; i++) {
		//			isVerify = true;
		//			indexSeparator = selectedTokens.get(i);
		//			for(int r = 0; r < nrows; r++) {
		//				separator = verifyRowWithIndexDelim(r, indexSeparator, labelIndex.get(i), firstColIndex);
		//				if(separator == null) {
		//					isVerify = false;
		//					break;
		//				}
		//			}
		//		}
		//		if(isVerify) {
		//			return new CustomProperties(CustomProperties.GRPattern.Regular, separator, indexSeparator);
		//		}
		//		else
		//			return null;
	}

	private String verifyRowWithIndexDelim(int rowID, String indexDelim, int labelIndex, int firstColIndex) {
		RawRow row = sampleRawRows.get(rowID);
		row = row.getResetClone();
		for(int c = ncols - 1; c >= 0; c--) {
			if(mapRow[rowID][c] != -1) {
				ValueTrimFormat.NumberTrimFormat vtfColIndex = new ValueTrimFormat.NumberTrimFormat(c + firstColIndex);
				ValueTrimFormat.StringTrimFormat vtfIndexDelim = new ValueTrimFormat.StringTrimFormat(indexDelim);
				ValueTrimFormat vtfColValue = VTF[rowID][c];

				Pair<Integer, Integer> pairCol;
				Pair<Integer, Integer> pairDelim;
				Pair<Integer, Integer> pairValue;
				if(labelIndex - firstColIndex != c) {

					// check Index text
					pairCol = row.findValue(vtfColIndex, false);
					if(pairCol.getKey() == -1)
						return null;

					// check the delimiter text
					pairDelim = row.findValue(vtfIndexDelim, true);
					if(pairDelim.getKey() == -1)
						return null;
					// check the value text
					pairValue = row.findValue(vtfColValue, true);
					if(pairValue.getKey() == -1)
						return null;
				}
				else {
					// check the just label text
					pairValue = row.findValue(vtfColValue, false);
					if(pairValue.getKey() == -1)
						return null;
				}
			}
		}
		HashSet<String> separators = row.getDelimsSet().getKey();
		String separator = separators.size() == 1 ? separators.iterator().next() : null;

		if(separator == null)
			return null;

		return separator;
	}

	private boolean isMapRowColValue(RawRow rawrow, int row, int col, ValueTrimFormat value) {
		ValueTrimFormat.NumberTrimFormat vtfRow = new ValueTrimFormat.NumberTrimFormat(row);
		ValueTrimFormat.NumberTrimFormat vtfCol = new ValueTrimFormat.NumberTrimFormat(col);
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
			if(rawrow.findValue(vtf, true).getKey() == -1) {
				mapped = false;
				break;
			}
		}
		return mapped;

	}

	private HashSet<String> getColIndexValueMappedTokens(RawRow rawrow, int col, ValueTrimFormat value) {
		ValueTrimFormat.NumberTrimFormat vtfColIndex = new ValueTrimFormat.NumberTrimFormat(col);
		ValueTrimFormat vtfColValue = value.getACopy();

		Pair<Integer, Integer> pairCol;
		Pair<Integer, Integer> pairValue;
		HashSet<String> tokens = new HashSet<>();
		RawRow row = rawrow.getResetClone();
		int lastIndex = 0;
		do {
			row.resetReserved();
			row.setLastIndex(lastIndex);
			pairCol = row.findValue(vtfColIndex, false);
			if(pairCol.getKey() == -1)
				break;

			lastIndex = row.getNumericLastIndex();

			pairValue = row.findValue(vtfColValue, true);
			if(pairValue.getKey() == -1)
				break;

			String t = row.getRaw().substring(pairCol.getKey() + pairCol.getValue(), pairValue.getKey());
			if(t.length() > 0)
				tokens.add(t);
			//lastIndex = pairCol.getKey()+1;
		}
		while(true);

		return tokens;
	}

	public int[][] getMapRow() {
		return mapRow;
	}

	public int[][] getMapCol() {
		return mapCol;
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
