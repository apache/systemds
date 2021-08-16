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

import com.google.gson.Gson;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.*;

public class GenerateReaderm {

	public static MatrixReader generateReader(String sampleRaw, MatrixBlock sampleMatrix) throws Exception {
		MatrixReader reader = null;

		ReaderMapping rp = new ReaderMapping(sampleRaw, sampleMatrix);

		boolean isMapped = rp.isMapped();
		if(!isMapped) {
			throw new Exception("Sample raw data and sample matrix don't match !!");
		}

		int nrows = sampleMatrix.getNumRows();
		int ncols = sampleMatrix.getNumColumns();
		long nnz = sampleMatrix.getNonZeros();

		//////////////////////////////////////////////////
		System.out.println("Mapped !!!!!!!!!!!!");
		Gson gson = new Gson();
		System.out.println("Map Row >> " + gson.toJson(rp.getMapRow()));
		System.out.println("Map Col >> " + gson.toJson(rp.getMapCol()));
		System.out.println("Map Size >> " + gson.toJson(rp.getMapSize()));

		boolean isRR = rp.isRowRegular();
		boolean isCR = false;

		// if Row is Regular so we need to check: is cols regular?
		if(isRR) {
			String[][] delims = rp.getDelimsOfRRCRMapping();
			System.out.println("Delims >> " + gson.toJson(delims));

			ArrayList<String> rowDelims = new ArrayList<>();
			Set<String> nullString = new HashSet<>();
			int maxSizeOfToken = 0;

			// append all delimiters as a string and then tokenize it
			for(int r = 0; r < nrows; r++) {
				StringBuilder sbRow = new StringBuilder();
				for(int c = 0; c < ncols + 1; c++) {
					sbRow.append(delims[r][c]);
					if(maxSizeOfToken == 0 || (delims[r][c].length() > 0 && delims[r][c].length() < maxSizeOfToken)) {
						maxSizeOfToken = delims[r][c].length();
					}
				}
				rowDelims.add(sbRow.toString());
			}

			String uniqueDelimiter = null;
			StringBuilder token = new StringBuilder();
			token.append(rowDelims.get(0).charAt(0));

			while(token.length() <= maxSizeOfToken) {
				boolean flagCurrToken = true;
				Set<String> ns = new HashSet<>();
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
					nullString = ns;
				}
				token.append(rowDelims.get(0).charAt(token.length()));
			}

			isCR = uniqueDelimiter != null;
			if(isCR) {
				System.out.println("********************************");
				//rcc = GenerateReader3.RCC.RRCR;
			}
			else {
				//rcc = GenerateReader3.RCC.RRCI;
				System.out.println("+++++++++++++++++++++++++++++++++++++++++++++++++++++ RR CI");
				ReaderMapping.ColIndexValue[][] de = rp.getDelimsOfRRCIMapping();
//				System.out.println(gson.toJson(de));
//				for(int r = 0; r < nrows; r++) {
//					for(int c = 0; c < ncols; c++) {
//						if(rp.getMapRow()[r][c] != -1)
//							System.out.print("[sep=(" + de[r][c].getSeparator() + ") , " + de[r][c].getIndSep() + "] ");
//					}
//					System.out.println();
//				}
			}
		}
		else {
			// check for Col value is unique
				/* example:
					MapRow: [ 1  2  3  4
				              5  6  7  5 ]
					MapCol: [10  12  10  12
					         10  11  13  10 ]
				*/
			int[][] mapRow = rp.getMapRow();
			int[] rowList = new int[(int) nnz];
			int minRow = -1;
			int index = 0;

			for(int r = 0; r < nrows; r++)
				for(int c = 0; c < ncols; c++) {
					int v = mapRow[r][c];
					if(v != -1) {
						if(minRow == -1)
							minRow = v;
						else if(minRow > v) {
							minRow = v;
						}
						rowList[index++] = mapRow[r][c];
					}
				}
			Arrays.sort(rowList);
			boolean isValueInARow = true;
			for(int i = 0; i < nnz; i++) {
				if(rowList[i] != minRow + i)
					isValueInARow = false;
			}

			if(isValueInARow) {
				ReaderMapping.RowColValue[] delims = rp.getDelimsOfRIMapping();
				for(int i = 0; i < nnz; i++) {
					System.out.println("type=" + delims[i].getType() + "[" + delims[i].getV0Delim() + "],[" + delims[i]
						.getV1Delim() + "],[" + delims[i].getV2Delim() + "]");
				}
			}
		}

		return reader;
	}
}
