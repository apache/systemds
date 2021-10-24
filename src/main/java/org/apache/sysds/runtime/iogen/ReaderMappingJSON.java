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
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public abstract class ReaderMappingJSON extends ReaderMapping {
	protected String[] mapCol;
	protected final ArrayList<String> finalMapCol;
	protected final ArrayList<FastJSONIndex> sampleRawRowsJSON;

	public ReaderMappingJSON(String raw) {
		sampleRawRowsJSON = new ArrayList<>();
		finalMapCol = new ArrayList<>();
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		try {
			while((value = br.readLine()) != null) {
				sampleRawRowsJSON.add(new FastJSONIndex(value));
			}
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
		finally {
			IOUtilFunctions.closeSilently(br);
			IOUtilFunctions.closeSilently(is);
		}
	}

	// Matrix Reader Mapping
	public static class MatrixReaderMapping extends ReaderMappingJSON {

		private MatrixBlock sampleMatrix;

		public MatrixReaderMapping(String raw, MatrixBlock matrix) {
			super(raw);
			this.sampleMatrix = matrix;
			nrows = sampleMatrix.getNumRows();
			ncols = sampleMatrix.getNumColumns();
			mapCol = new String[ncols];

			HashSet<String> names = new HashSet<>();

			for(FastJSONIndex fji : sampleRawRowsJSON) {
				names.addAll(fji.getNames());
			}
			String[] colNames = new String[names.size()];
			BitSet bitSet = new BitSet(colNames.length);
			MatrixBlock rawMatrix = new MatrixBlock(nrows, names.size(), false, -1);
			for(int r = 0; r < nrows; r++) {
				int c = 0;
				for(String k : names) {
					colNames[c] = k;
					rawMatrix.setValue(r, c++, sampleRawRowsJSON.get(r).getDoubleValue(k));
				}
			}

			// looking for the col map
			for(int c = 0; c < ncols; c++) {
				boolean flagColMap = false;
				for(int i = 0; i < colNames.length && !flagColMap; i++) {
					if(bitSet.get(i))
						continue;
					flagColMap = true;
					for(int r = 0; r < nrows; r++) {
						if(sampleMatrix.getValue(r, c) != rawMatrix.getValue(r, i)) {
							flagColMap = false;
							break;
						}
					}
					if(flagColMap) {
						finalMapCol.add(colNames[i]);
						bitSet.set(i);
					}
				}
			}
			mapCol = new String[finalMapCol.size()];
			for(int i = 0; i < finalMapCol.size(); i++) {
				mapCol[i] = finalMapCol.get(i);
			}
			mapped = mapCol.length > 0;
		}

		@Override public CustomProperties getFormatProperties() throws Exception {
			CustomProperties properties = new CustomProperties(mapCol);
			return properties;
		}
	}

	// Frame Reader Mapping
	public static class FrameReaderMapping extends ReaderMappingJSON {

		private FrameBlock sampleFrame;
		private Types.ValueType[] schema;
		private final ArrayList<Types.ValueType> finalSchema;

		public FrameReaderMapping(String raw, FrameBlock frame) {
			super(raw);
			finalSchema = new ArrayList<>();
			this.sampleFrame = frame;
			nrows = sampleFrame.getNumRows();
			ncols = sampleFrame.getNumColumns();
			schema = sampleFrame.getSchema();
			mapCol = new String[ncols];

			Map<String, Types.ValueType> names = new HashMap<>();
			for(FastJSONIndex fji : sampleRawRowsJSON) {
				names.putAll(fji.getNamesType());
			}
			String[] rawColNames = new String[names.size()];
			Types.ValueType[] rawSchema = new Types.ValueType[names.size()];
			BitSet bitSet = new BitSet(rawColNames.length);
			Object[][] data = new Object[nrows][rawColNames.length];

			for(int r = 0; r < nrows; r++) {
				int c = 0;
				FastJSONIndex fji = sampleRawRowsJSON.get(r);
				for(String k : names.keySet()) {
					rawColNames[c] = k;
					rawSchema[c] = names.get(k);
					data[r][c++] = fji.getObjectValue(k);
				}
			}
			// looking for the col map
			for(int c = 0; c < ncols; c++) {
				boolean flagColMap = false;
				for(int i = 0; i < rawColNames.length && !flagColMap; i++) {
					if(bitSet.get(i))
						continue;
					flagColMap = true;
					for(int r = 0; r < nrows; r++) {
						if(rawSchema[i].isNumeric() && schema[c].isNumeric()) {
							double in1 = data[r][i] != null ? UtilFunctions.getDouble(data[r][i]) : 0;
							double in2 = UtilFunctions.getDouble(sampleFrame.get(r, c));
							if(in1 != in2) {
								flagColMap = false;
								break;
							}
						}
						else if(!rawSchema[i].equals(schema[c])) {
							flagColMap = false;
							break;
						}

						else if(rawSchema[i] == Types.ValueType.STRING && schema[c] == Types.ValueType.STRING) {
							String in1 = UtilFunctions.objectToString(sampleFrame.get(r, c));
							String in2 = UtilFunctions.objectToString(data[r][i]);

							if(in2 != null && in2.trim().equals(""))
								in2 = null;
							if(in2 != null)
								in2 = in2.trim();

							if(UtilFunctions.compareTo(Types.ValueType.STRING, in1, in2) != 0) {
								flagColMap = false;
								break;
							}
						}
					}
					if(flagColMap) {
						bitSet.set(i);
						finalMapCol.add(rawColNames[i]);
						finalSchema.add(schema[c]);
					}
				}
			}
			mapCol = new String[finalMapCol.size()];
			schema = new Types.ValueType[mapCol.length];
			for(int i = 0; i < finalMapCol.size(); i++) {
				mapCol[i] = finalMapCol.get(i);
				schema[i] = finalSchema.get(i);
			}
			mapped = mapCol.length > 0;
		}

		@Override public CustomProperties getFormatProperties() throws Exception {
			CustomProperties properties = new CustomProperties(mapCol, schema);
			return properties;
		}
	}
}
