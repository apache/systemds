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

package org.apache.sysds.runtime.io;

import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.*;
import java.util.*;

public class GenerateReader {

	private static final IDSequence CLASS_ID = new IDSequence();

	public GenerateReader() {}

	public static MatrixReader generateReader(String stream, MatrixBlock sample) throws IOException {

		MatrixReader reader = null;
		FileFormatProperties ffp = formatInference(stream);
		String fileFormatPropertiesString = "new ";

		if(ffp instanceof FileFormatPropertiesCSV) {
			FileFormatPropertiesCSV ffpCSV = (FileFormatPropertiesCSV) ffp;
			reader = new ReaderTextCSV(ffpCSV);
			fileFormatPropertiesString += "FileFormatPropertiesCSV(" + ffpCSV.hasHeader() + ",\"" + ffpCSV
				.getDelim() + "\"," + ffpCSV.isSparse() + ")";
		}
		else if(ffp instanceof FileFormatPropertiesLIBSVM) {
			FileFormatPropertiesLIBSVM ffpLIBSVM = (FileFormatPropertiesLIBSVM) ffp;
			reader = new ReaderTextLIBSVM(ffpLIBSVM);
			fileFormatPropertiesString += "FileFormatPropertiesLIBSVM(" + ffpLIBSVM.getDelim() + "," + ffpLIBSVM
				.getIndexDelim() + ")";
		}

		int rlen = 0, clen = 0;
		String value;
		InputStream is = IOUtilFunctions.toInputStream(stream);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		while((value = br.readLine()) != null) //foreach line
		{
			if(ffp instanceof FileFormatPropertiesCSV) {
				if(clen == 0) {
					clen = IOUtilFunctions.splitCSV(value, ((FileFormatPropertiesCSV) ffp).getDelim()).length;
					if(((FileFormatPropertiesCSV) ffp).hasHeader()) {
						rlen--;
					}
				}
			}
			else if(ffp instanceof FileFormatPropertiesLIBSVM) {
				String items[] = IOUtilFunctions.splitCSV(value, ((FileFormatPropertiesCSV) ffp).getDelim());
				for(int i = 1; i < items.length; i++) {
					String cell = IOUtilFunctions
						.splitCSV(items[i], ((FileFormatPropertiesLIBSVM) ffp).getIndexDelim())[0];
					int ci = UtilFunctions.parseToInt(cell);
					if(clen < ci) {
						clen = ci;
					}
				}
			}
			rlen++;
		}
		is = IOUtilFunctions.toInputStream(stream);
		MatrixBlock mbStream = reader.readMatrixFromInputStream(is, rlen, clen, -1, -1);

		Map<Integer, Integer> match = match(mbStream, sample);
		String baseClassName = reader.getClass().getSimpleName();
		String fileFormatProperties = ffp.getClass().getSimpleName();
		reader = generate(baseClassName, fileFormatProperties, fileFormatPropertiesString, match);

		return reader;
	}

	private static Map<Integer, Integer> match(MatrixBlock stream, MatrixBlock sample) {

		int rows = stream.getNumRows();
		int cols = stream.getNumColumns();
		int srows = sample.getNumRows();
		int scols = sample.getNumColumns();

		ArrayList<Map<Integer, Integer>> matchList = new ArrayList<>();

		for(int i = 0; i <= rows - srows; i++) {
			for(int j = 0; j <= cols - scols; j++) {
				MatrixBlock slice1 = stream.slice(i, i + srows - 1, j, j + scols - 1);
				Map<Integer, Integer> matchItem = findSliceMatches(slice1, sample);
				if(matchItem.size() == scols) {
					matchList.add(matchItem);
				}
			}
		}

		if(matchList.size() == 0) {
			return null;
		}
		else if(matchList.size() == 1) {
			return matchList.get(0);
		}
		else {
			// Find the best matching between all matches
			//check the matches are unique or duplicate
			Map<Integer, Set<Integer>> matchSet = new HashMap<>();
			for(Map<Integer, Integer> item : matchList) {
				for(Integer c : item.keySet()) {
					Set<Integer> tmp = matchSet.get(c);
					if(tmp == null)
						tmp = new HashSet<>();
					tmp.add(item.get(c));
				}
			}
			// a: unique check
			boolean isUnique = true;
			for(Integer k : matchSet.keySet()) {
				if(matchSet.get(k).size() != 1)
					isUnique = false;
			}
			// if the col matches is unique then return the first index
			if(isUnique) {
				return matchList.get(0);
			}
			// TODO: if the col matches are not unique find the best order
			else {
				return matchList.get(0);
			}
		}
	}

	private static Map<Integer, Integer> findSliceMatches(MatrixBlock slice1, MatrixBlock slice2) {

		int nrows = slice1.getNumRows();
		int ncols = slice1.getNumColumns();
		Map<Integer, Integer> colMatches = new HashMap<>();

		DenseBlock slice1db = slice1.getDenseBlock();
		DenseBlock slice2db = slice2.getDenseBlock();

		for(int i = 0; i < ncols; i++) {
			for(int j = 0; j < ncols; j++) {
				boolean rowMatch = true;
				for(int k = 0; k < nrows; k++) {
					if(slice1db.get(k, i) != slice2db.get(k, j)) {
						rowMatch = false;
						break;
					}
				}
				if(rowMatch) {
					colMatches.put(i, j);
				}
			}
		}
		return colMatches;
	}

	private static FileFormatProperties formatInference(String stream) throws IOException {

		FileFormatProperties ffp;

		// what is the stream format? check for libsvm, csv, text, binary, ...
		InputStream is = IOUtilFunctions.toInputStream(stream);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		ArrayList<Character> delimiters = null;

		String value;
		boolean firstLineFlag = true;
		String firstLineString = null;
		while((value = br.readLine()) != null) //foreach line
		{
			ArrayList<Character> characters = new ArrayList<>();
			for(char character : value.toCharArray()) {
				boolean flag = Character.isDigit(character) | character == '.';
				if(!flag) {
					characters.add(character);
				}
			}
			if(!firstLineFlag) {
				delimiters.retainAll(characters);
			}
			else {
				delimiters = characters;
				firstLineString = value;
				firstLineFlag = false;
			}
		}
		Set<Character> delimiterSet = new HashSet<>();
		delimiterSet.addAll(delimiters);

		switch(delimiterSet.size()) {
			//0: for one column datasets
			//1: one delimiter just valid for CSV datasets
			// CSV file format detected
			case 0:
			case 1:
				String delim = delimiterSet.size() > 0 ? delimiterSet.iterator().next()
					.toString() : DataExpression.DEFAULT_DELIM_DELIMITER;

				// look for the header: has header?
				// split the first line and check the elements are digit or string
				String firstLineCols[] = IOUtilFunctions.splitCSV(firstLineString, delim);
				boolean hasHeader = false;
				for(String s : firstLineCols) {
					try {
						Double.parseDouble(s);
					}
					catch(NumberFormatException exception) {
						hasHeader = true;
						break;
					}
				}
				ffp = new FileFormatPropertiesCSV(hasHeader, delim, false);
				break;
			// for two delimiters we can check the libsvm format
			case 2:
				// recognize the delim and indexDelim
				Iterator<Character> it = delimiterSet.iterator();
				delim = it.next().toString();
				String indexDelim = it.next().toString();
				String firstLineItems1[] = IOUtilFunctions.splitCSV(firstLineString, delim);
				String firstLineItems2[] = IOUtilFunctions.splitCSV(firstLineString, indexDelim);
				if(firstLineItems1[0].length() > firstLineItems2[0].length()) {
					delim = delim + indexDelim;
					indexDelim = delim.substring(0, (delim.length() - indexDelim.length()));
					delim = delim.substring(indexDelim.length());
				}
				ffp = new FileFormatPropertiesLIBSVM(delim, indexDelim);
				break;
			default:
				ffp = new FileFormatProperties();
				break;
		}
		return ffp;
	}

	public static MatrixReader generate(String baseClassName, String fileFormatProperties,
		String fileFormatPropertiesString, Map<Integer, Integer> match) throws IOException {
		String cname = "MatrixReader" + CLASS_ID.getNextID();

		String keys = "";
		String values = "";
		for(Integer k : match.keySet()) {
			Integer v = match.get(k);
			if(k != v) {
				keys += k + ",";
				values += v + ",";
			}
		}
		if(keys.length() > 0) {
			keys = keys.substring(0, keys.length() - 1);
			values = values.substring(0, values.length() - 1);
		}

		StringBuilder sb = new StringBuilder();
		sb.append("import org.apache.sysds.runtime.io." + baseClassName + ";\n");
		sb.append("import org.apache.sysds.runtime.io." + fileFormatProperties + ";\n");
		sb.append("import org.apache.sysds.runtime.DMLRuntimeException;\n");
		sb.append("import org.apache.sysds.runtime.matrix.data.MatrixBlock;\n");
		sb.append("import java.io.IOException;\n");
		sb.append("import java.io.InputStream;\n");
		sb.append("public class " + cname + " extends " + baseClassName + "{\n");
		sb.append("private static final int keys[] = {" + keys + "};\n");
		sb.append("private static final int values[] = {" + values + "};\n");

		sb.append("public " + cname + "() {super(" + fileFormatPropertiesString + ");}\n");
		sb.append("	public " + cname + "(" + fileFormatProperties + " props) {\n");
		sb.append("		super(props);\n");
		sb.append("	}\n");

		sb.append(
			"	@Override public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)\n");
		sb.append("		throws IOException, DMLRuntimeException {\n");
		sb.append("MatrixBlock mb = super.readMatrixFromHDFS(fname, rlen, clen, blen, estnnz);\n");
		sb.append("		return updateMatrixBlock(super.readMatrixFromHDFS(fname, rlen, clen, blen, estnnz));\n");
		sb.append("	}\n");
		sb.append(
			"	@Override public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)\n");
		sb.append("		throws IOException, DMLRuntimeException {\n");
		sb.append("		return updateMatrixBlock(super.readMatrixFromInputStream(is, rlen, clen, blen, estnnz));\n");
		sb.append("	}\n");
		sb.append("private static MatrixBlock updateMatrixBlock(MatrixBlock mb){\n");
		sb.append("		for(int r= 0;r<mb.getNumRows();r++){\n");
		sb.append("			for(int i=0;i<keys.length;i++){\n");
		sb.append("				double temp = mb.getValue(r,i);\n");
		sb.append("				mb.setValue(r,keys[i],mb.getValue(r,values[i]));\n");
		sb.append("				mb.setValue(r,values[i], temp);\n");
		sb.append("			}\n");
		sb.append("		}\n");
		sb.append("		return mb;\n");
		sb.append("	}\n");
		sb.append("}\n");

		// compile class, and create MatrixReader object
		try {
			MatrixReader mr = (MatrixReader) CodegenUtils.compileClass(cname, sb.toString()).newInstance();
			return mr;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to compile MatrixReader.", e);
		}
	}
}
