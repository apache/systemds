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
import org.apache.sysds.runtime.util.UtilFunctions;

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
	private boolean fullMap;
	private boolean upperTriangularMap;
	private boolean lowerTriangularMap;
	private boolean symmetricMap;
	private boolean skewSymmetricMap;
	private boolean patternMap;
	private Object patternValueMap;
	private Types.ValueType patternValueType;

	private final int nrows;
	private final int ncols;
	private int nlines;
	private int NaN;
	private ArrayList<RawIndex> sampleRawIndexes;
	private MatrixBlock sampleMatrix;
	private FrameBlock sampleFrame;
	private Types.ValueType[] schema;
	private final boolean isMatrix;

	public ReaderMapping(int nlines, int nrows, int ncols, ArrayList<RawIndex> sampleRawIndexes, MatrixBlock matrix) throws Exception {
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
		NaN = 0;

		// Set "-1" as default value for all defined matrix
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				mapRow[r][c] = mapCol[r][c] = mapLen[r][c] = -1;

		int itRow = 0;
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(isIndexMapping || ((this.isMatrix && this.sampleMatrix.getValue(r, c) != 0) || (!this.isMatrix && ((!schema[c].isNumeric() && this.sampleFrame.get(r,
					c) != null) || (schema[c].isNumeric() && this.sampleFrame.getDouble(r, c) != 0))))) {
					HashSet<Integer> checkedLines = new HashSet<>();
					while(checkedLines.size() < nlines) {
						RawIndex ri = sampleRawIndexes.get(itRow);
						Pair<Integer, Integer> pair = this.isMatrix ? ri.findValue(sampleMatrix.getValue(r, c)) : ri.findValue(sampleFrame.get(r, c), schema[c]);
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
				else
					NaN++;
			}
		}

		// analysis mapping of values
		// 1. check (exist, partially exist, not exist)
		// 2. check the records represented in single/multilines
		// 3. check the Symmetric, Skew-Symmetric, Pattern, and Array

		int fullMap = 0;
		int upperTriangular = 0;
		int upperTriangularZeros = 0;
		int lowerTriangular = 0;
		int lowerTriangularZeros = 0;
		boolean singleLineRecord = true;

		// check full map
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				if(mapRow[r][c] != -1)
					fullMap++;

		// check for upper and lower triangular
		if(nrows == ncols) {
			this.upperTriangularMap = true;
			this.lowerTriangularMap = true;
			this.symmetricMap = true;
			this.skewSymmetricMap = true;
			this.patternMap = false;
			this.patternValueMap = null;

			// pattern check for Frame: in Frame the schema must be same for all columns
			boolean homoSchema = true;
			Types.ValueType vtc0 = null;
			if(!this.isMatrix) {
				vtc0 = this.sampleFrame.getSchema()[0];
				for(int c = 1; c < ncols && homoSchema; c++)
					homoSchema = this.sampleFrame.getSchema()[c] == vtc0;
			}

			for(int r = 0; r < nrows; r++) {
				// upper triangular check
				for(int c = r; c < ncols && this.upperTriangularMap; c++)
					if(this.checkValueIsNotNullZero(r, c) && mapRow[r][c] == -1)
						this.upperTriangularMap = false;

				for(int c = 0; c < r && this.upperTriangularMap; c++)
					if(this.checkValueIsNotNullZero(r, c))
						this.upperTriangularMap = false;

				// lower triangular check
				for(int c = 0; c <= r && this.lowerTriangularMap; c++)
					if(this.checkValueIsNotNullZero(r, c) && mapRow[r][c] == -1)
						this.lowerTriangularMap = false;

				for(int c = r + 1; c < ncols && this.lowerTriangularMap; c++)
					if(this.checkValueIsNotNullZero(r, c))
						this.lowerTriangularMap = false;

				// Symmetric check
				for(int c = 0; c <= r && this.symmetricMap; c++)
					this.symmetricMap = this.checkSymmetricValue(r, c, 1);

				// Skew-Symmetric check
				for(int c = 0; c <= r && this.skewSymmetricMap; c++)
					this.skewSymmetricMap = this.checkSymmetricValue(r, c, -1);

				// pattern check for Matrix
				if(this.isMatrix) {
					HashSet<Double> patternValueSet = new HashSet<>();
					for(int c = 0; c < ncols; c++)
						patternValueSet.add(this.sampleMatrix.getValue(r, c));
					if(patternValueSet.size() == 1) {
						this.patternValueType = Types.ValueType.FP64;
						this.patternMap = true;
						this.patternValueMap = patternValueSet.iterator().next();
					}
				}
				else {
					if(homoSchema) {
						HashSet<Object> patternValueSet = new HashSet<>();
						for(int c = 0; c < ncols; c++)
							patternValueSet.add(this.sampleFrame.get(r, c));
						if(patternValueSet.size() == 1) {
							this.patternValueType = vtc0;
							this.patternMap = true;
							this.patternValueMap = patternValueSet.iterator().next();
						}
					}
				}
			}
		}

		System.out.println("upperTriangularMap=" + upperTriangularMap);
		System.out.println("lowerTriangularMap=" + lowerTriangularMap);
		System.out.println("symmetric=" + symmetricMap);
		System.out.println("skewSymmetricMap = " + skewSymmetricMap);
		System.out.println("patternMap=" + patternMap);
		System.out.println("patternValueType = "+patternValueType);
		System.out.println("patternValueMap=" + UtilFunctions.objectToString(patternValueType));


		return false;
	}

	private boolean checkValueIsNotNullZero(int r, int c) {
		boolean result;
		if(this.isMatrix)
			result = this.sampleMatrix.getValue(r, c) != 0;
		else {
			if(this.sampleFrame.getSchema()[c].isNumeric())
				result = this.sampleFrame.getDouble(r, c) != 0;
			else
				result = this.sampleFrame.get(r, c) != null;
		}
		return result;
	}

	// Symmetric checks just available for numeric values in the frame representations
	private boolean checkSymmetricValue(int r, int c, int a) {
		boolean result;
		if(this.isMatrix)
			result = this.sampleMatrix.getValue(r, c) == this.sampleMatrix.getValue(c, r) * a;
		else if(this.sampleFrame.getSchema()[c].isNumeric())
			result = this.sampleFrame.getDouble(r, c) == this.sampleFrame.getDouble(c, r) * a;
		else
			result = this.sampleFrame.get(r, c).equals(this.sampleFrame.get(c, r));

		return result;
	}

	public int getNaN() {
		return NaN;
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
