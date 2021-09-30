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

import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;

public abstract class ReaderMappingJSON extends ReaderMapping {
	protected String[] mapCol;
	protected final ArrayList<RawRowJSON> sampleRawRowsJSON;

	public ReaderMappingJSON(String raw){
		sampleRawRowsJSON = new ArrayList<>();
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		try {
			while((value = br.readLine()) != null) {
				sampleRawRowsJSON.add(new RawRowJSON(value));
			}
		}
		catch(Exception e){
			throw new RuntimeException(e);
		}
		finally {
			IOUtilFunctions.closeSilently(br);
			IOUtilFunctions.closeSilently(is);
		}
	}

	@Override public CustomProperties getFormatProperties() throws Exception {
		CustomProperties properties = new CustomProperties(mapCol);
		return properties;
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

			for(RawRowJSON rrj : sampleRawRowsJSON) {
				names.addAll(rrj.getSchemaNames());
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
						mapCol[c] = colNames[i];
						bitSet.set(i);
					}
				}
			}
			// verify the mapped
			int sum = 0;
			for(int i = 0; i < bitSet.length(); i++)
				if(bitSet.get(i))
					sum++;

			mapped = sum == ncols;
		}
	}

	// Matrix Reader Mapping
	public static class FrameReaderMapping extends ReaderMappingJSON {

		private FrameBlock sampleFrame;

		public FrameReaderMapping(String raw, FrameBlock frame) {
			super(raw);
		}
	}
}
