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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import java.util.ArrayList;
import java.util.Stack;

public abstract class ReaderMappingJSON {
	protected String[] mapCol;
	protected static int nrows;
	protected static int ncols;
	protected boolean mapped;
	protected final ArrayList<RawRowJSON> sampleRawRowsJSON;

	public ReaderMappingJSON(String raw) {
		sampleRawRowsJSON = new ArrayList<>();
		Stack<Character> stack = new Stack<>();

		// remove "Enter" from input data
		String rawJSON = raw.trim().replace("\n", "").replace("\r", "");
		StringBuilder row = new StringBuilder();
		for(Character ch : rawJSON.toCharArray()) {
			row.append(ch);
			if(ch.equals('{')) {
				stack.push(ch);
			}
			else if(ch.equals('}')) {
				stack.pop();
				if(stack.size() == 0) {
					sampleRawRowsJSON.add(new RawRowJSON(row.toString()));
					row = new StringBuilder();
				}
			}
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

//			for(int c = 0; c < ncols; c++) {
//				boolean flagCollMap = true;
//				for(int r = 0; r < nrows; r++) {
//
//				}
//			}
		}
	}

	protected void runMapping() {

	}
}
