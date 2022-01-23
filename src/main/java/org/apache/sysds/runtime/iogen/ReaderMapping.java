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
import java.util.ArrayList;
import java.util.HashSet;

public class ReaderMapping {

	private int[][] mapRow;
	private int[][] mapCol;
	private int[][] mapLen;
	private boolean mapped;
	private final int nrows;
	private final int ncols;
	private int nlines;
	private ArrayList<RawIndex> sampleRawIndexes;
	private MatrixBlock sampleMatrix;
	private FrameBlock sampleFrame;
	private Types.ValueType[] schema;
	private final boolean isMatrix;

	public ReaderMapping(int nlines, int nrows, int ncols, ArrayList<RawIndex> sampleRawIndexes, MatrixBlock matrix)
		throws Exception {
		this.nlines = nlines;
		this.nrows = nrows;
		this.ncols = ncols;
		this.sampleRawIndexes = sampleRawIndexes;
		this.sampleMatrix = matrix;
		this.isMatrix = true;
		this.runMapping(true);
	}

	public ReaderMapping(String raw, MatrixBlock matrix) throws Exception {
		this.ReadRaw(raw);
		this.isMatrix = true;
		this.sampleMatrix = matrix;
		this.nrows = this.sampleMatrix.getNumRows();
		this.ncols = this.sampleMatrix.getNumColumns();
		this.runMapping(false);
	}

	public ReaderMapping(String raw, FrameBlock frame) throws Exception {
		this.ReadRaw(raw);
		this.isMatrix = false;
		this.sampleFrame = frame;
		this.nrows = this.sampleFrame.getNumRows();
		this.ncols = this.sampleFrame.getNumColumns();
		this.schema = this.sampleFrame.getSchema();
		this.runMapping(false);
	}

	private void ReadRaw(String raw) throws Exception {
		this.sampleRawIndexes = new ArrayList<>();
		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		int nlines = 0;

		while((value = br.readLine()) != null) {
			this.sampleRawIndexes.add(new RawIndex(value));
			nlines++;
		}
		this.nlines = nlines;
	}

	private void runMapping(boolean isIndexMapping) {
		mapped = findMapping(isIndexMapping);
	}

	protected boolean findMapping(boolean isIndexMapping) {
		mapRow = new int[nrows][ncols];
		mapCol = new int[nrows][ncols];
		mapLen = new int[nrows][ncols];

		// Set "-1" as default value for all defined matrix
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				mapRow[r][c] = mapCol[r][c] = mapLen[r][c] = -1;

		int itRow = 0;
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(isIndexMapping || ((this.isMatrix && this.sampleMatrix.getValue(r, c) != 0) || (!this.isMatrix && this.sampleFrame.get(
					r, c) != null))) {
					HashSet<Integer> checkedLines = new HashSet<>();
					while(checkedLines.size() < nlines) {
						RawIndex ri = sampleRawIndexes.get(itRow);
						Pair<Integer, Integer> pair = this.isMatrix ? ri.findValue(
							sampleMatrix.getValue(r, c)) : ri.findValue(sampleFrame.get(r, c), schema[c]);
						if(pair != null) {
							mapRow[r][c] = itRow;
							mapCol[r][c] = pair.getKey();
							mapLen[r][c] = pair.getValue();
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
		}
		boolean flagMap = true;
		for(int r = 0; r < nrows && flagMap; r++)
			for(int c = 0; c < ncols && flagMap; c++)
				if(mapRow[r][c] == -1 && ((!this.isMatrix && this.sampleFrame.get(r,
					c) != null) || (this.isMatrix && this.sampleMatrix.getValue(r, c) != 0))) {
					flagMap = false;
				}
		return flagMap;
	}



	public int[][] getMapRow() {
		return mapRow;
	}

	public int[][] getMapCol() {
		return mapCol;
	}

	public int[][] getMapLen() {
		return mapLen;
	}

	public ArrayList<RawIndex> getSampleRawIndexes() {
		return sampleRawIndexes;
	}

	public int getNrows() {
		return nrows;
	}

	public int getNcols() {
		return ncols;
	}

	public int getNlines() {
		return nlines;
	}

	public boolean isMapped() {
		return mapped;
	}
}
