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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ReaderMappingIndex {
	private int[][] mapRow;
	private int[][] mapCol;
	private int[][] mapLen;
	private MappingProperties mappingProperties;
	private final int nrows;
	private final int ncols;
	private int nlines;
	private int actualValueCount;
	private RawIndex[] sampleRawIndexes;
	private MatrixBlock sampleMatrix;
	private FrameBlock sampleFrame;
	private Types.ValueType[] schema;
	private final boolean isMatrix;
	public ReaderMappingIndex(int nlines, int nrows, int ncols, RawIndex[] sampleRawIndexes, MatrixBlock matrix) throws Exception {
		this.nlines = nlines;
		this.nrows = nrows;
		this.ncols = ncols;
		this.sampleRawIndexes = sampleRawIndexes;
		this.sampleMatrix = matrix;
		this.isMatrix = true;
		//this.runMapping(true);
	}

	public ReaderMappingIndex(String raw, MatrixBlock matrix) throws Exception {
		this.ReadRaw(raw);
		this.isMatrix = true;
		this.sampleMatrix = matrix;
		this.nrows = this.sampleMatrix.getNumRows();
		this.ncols = this.sampleMatrix.getNumColumns();
		//this.runMapping(false);
	}

	public ReaderMappingIndex(String raw, FrameBlock frame) throws Exception {
		this.ReadRaw(raw);
		this.isMatrix = false;
		this.sampleFrame = frame;
		this.nrows = this.sampleFrame.getNumRows();
		this.ncols = this.sampleFrame.getNumColumns();
		this.schema = this.sampleFrame.getSchema();
		//this.runMapping(false);
	}

	private void ReadRaw(String raw) throws Exception {

		InputStream is = IOUtilFunctions.toInputStream(raw);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;

		ArrayList<String> rawList = new ArrayList<>();
		while((value = br.readLine()) != null) {
			rawList.add(value);
		}
		this.nlines = rawList.size();
		this.sampleRawIndexes = new RawIndex[this.nlines];
		int numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		try {
			ExecutorService pool = CommonThreadPool.get(numThreads);
			ArrayList<BuildRawIndexTask> tasks = new ArrayList<>();
			int blklen = (int) Math.ceil((double) nlines / numThreads);
			for(int i = 0; i < numThreads; i++) {
				int beginIndex = i * blklen;
				int endIndex = Math.min((i + 1) * blklen, nlines);
				tasks.add(new BuildRawIndexTask(rawList, this.sampleRawIndexes, beginIndex, endIndex));
			}
			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();

			//check for exceptions
			for(Future<Object> task : rt)
				task.get();
		}
		catch(Exception e) {
			throw new RuntimeException("Failed parallel ReadRaw.", e);
		}

	}

	private void runMapping(boolean isIndexMapping) {

		this.mapRow = new int[nrows][ncols];
		this.mapCol = new int[nrows][ncols];
		this.mapLen = new int[nrows][ncols];
		this.mappingProperties = new MappingProperties();

		// Set "-1" as default value for all defined matrix
		for(int r = 0; r < nrows; r++)
			for(int c = 0; c < ncols; c++)
				this.mapRow[r][c] = this.mapCol[r][c] = this.mapLen[r][c] = -1;

		int itRow = 0;
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(isIndexMapping || checkValueIsNotNullZero(r, c)) {
					HashSet<Integer> checkedLines = new HashSet<>();
					while(checkedLines.size() < nlines) {
						RawIndex ri = this.sampleRawIndexes[itRow];
						Pair<Integer, Integer> pair = this.isMatrix ? ri.findValue(this.sampleMatrix.getValue(r, c)) : ri.findValue(this.sampleFrame.get(r, c), this.schema[c]);
						if(pair != null) {
							this.mapRow[r][c] = itRow;
							this.mapCol[r][c] = pair.getKey();
							this.mapLen[r][c] = pair.getValue();
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

		// analysis mapping of values
		// 1. check (exist, partially exist, not exist)
		actualValueCount = 0;
		int mappedValueCount = 0;
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(checkValueIsNotNullZero(r, c)) {
					actualValueCount++;
					if(this.mapRow[r][c] != -1)
						mappedValueCount++;
				}
			}
		}
		if(actualValueCount == mappedValueCount) {
			this.mappingProperties.setTypicalRepresentation();
			this.mappingProperties.setDataFullExist();
		}
		else if(actualValueCount > 0 && mappedValueCount == 0)
			this.mappingProperties.setDataNotExist();

		else if(mappedValueCount > 0 && mappedValueCount < actualValueCount)
			this.mappingProperties.setDataPartiallyExist();

		// 2. check the records represented in single/multilines
		boolean singleLine = true;

		// first mapped value
		int[] firstLineNumbers = new int[nrows];
		for(int r = 0; r < nrows; r++) {
			int c = 0;
			firstLineNumbers[r] = -1;
			for(; c < ncols && firstLineNumbers[r] == -1; c++)
				firstLineNumbers[r] = mapRow[r][c];
			// other mapped 
			for(; c < ncols && singleLine; c++)
				if(mapRow[r][c] != -1)
					singleLine = firstLineNumbers[r] == mapRow[r][c];

		}
		for(int r=0; r<nrows-1 && singleLine; r++)
				singleLine = firstLineNumbers[r+1] - firstLineNumbers[r] == 1;

		if(singleLine) {
			mappingProperties.setRecordSingleLine();
			// 3.a check for array representation
			boolean allValuesInALine = true;
			for(int r=0; r<nrows && allValuesInALine; r++){
				for(int c=0; c<ncols; c++){
					if(mapRow[r][c] != -1 && mapRow[r][c] != firstLineNumbers[0]) {
						allValuesInALine = false;
						break;
					}
				}
			}

			// when all values are continuously are in a single line it is an Array representation
			if(allValuesInALine){
				// check the Array is in Row or Col wise
				int t = 0;
				for(int c = 0; c<ncols; c++){
					for(int r=0; r<nrows-1; r++){
						if(mapCol[r][c] != -1)
							continue;
						if(mapCol[r][c] > mapCol[r+1][c])
							t++;
					}
				}

				if((float)t/actualValueCount <0.03)
					this.mappingProperties.setArrayRowWiseRepresentation();
				else
					this.mappingProperties.setArrayColWiseRepresentation();
			}
		}
		else {
			mappingProperties.setRecordMultiLine();
			// 3.a check for array representation
			// TODO: array properties for multi-line
		}

		// 3.b check the Typical, Symmetric, Skew-Symmetric, Pattern, and Array
		// check for upper and lower triangular
		if(nrows == ncols && !this.mappingProperties.isRepresentation()) {
			boolean symmetricMap = true;

			// Symmetric check
			for(int r = 0; r < nrows && symmetricMap; r++) {
				for(int c = 0; c <= r && symmetricMap; c++)
					symmetricMap = this.checkSymmetricValue(r, c, 1);
			}
			if(symmetricMap)
				mappingProperties.setSymmetricRepresentation();

			// Skew-Symmetric check
			if(!mappingProperties.isRepresentation()) {
				boolean skewSymmetricMap = true;
				for(int r = 0; r < nrows && skewSymmetricMap; r++) {
					for(int c = 0; c <= r && skewSymmetricMap; c++)
						skewSymmetricMap = this.checkSymmetricValue(r, c, -1);
				}
				if(skewSymmetricMap)
					mappingProperties.setSkewSymmetricRepresentation();
			}

			// Pattern check
			if(!mappingProperties.isRepresentation()) {
				boolean patternMap = false;
				Object patternValueMap = null;

				// pattern check for Frame: in Frame the schema must be same for all columns
				boolean homoSchema = true;
				Types.ValueType vtc0 = null;
				if(!this.isMatrix) {
					vtc0 = this.sampleFrame.getSchema()[0];
					for(int c = 1; c < ncols && homoSchema; c++)
						homoSchema = this.sampleFrame.getSchema()[c] == vtc0;
				}
				// pattern check for Matrix representation
				for(int r = 0; r < nrows; r++) {
					if(this.isMatrix) {
						HashSet<Double> patternValueSet = new HashSet<>();
						for(int c = 0; c < ncols; c++)
							patternValueSet.add(this.sampleMatrix.getValue(r, c));
						if(patternValueSet.size() == 1) {
							vtc0 = Types.ValueType.FP64;
							patternMap = true;
							patternValueMap = patternValueSet.iterator().next();
						}
					}
					else { // pattern check for Frame representation
						if(homoSchema) {
							HashSet<Object> patternValueSet = new HashSet<>();
							for(int c = 0; c < ncols; c++)
								patternValueSet.add(this.sampleFrame.get(r, c));
							if(patternValueSet.size() == 1) {
								patternMap = true;
								patternValueMap = patternValueSet.iterator().next();
							}
						}
					}
				}

				if(patternMap)
					mappingProperties.setPatternRepresentation(vtc0, patternValueMap);
			}
		}
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

	public int[][] getMapRow() {
		return mapRow;
	}

	public int[][] getMapCol() {
		return mapCol;
	}

	public int[][] getMapLen() {
		return mapLen;
	}

	public RawIndex[] getSampleRawIndexes() {
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

	public MappingProperties getMappingProperties() {
		return mappingProperties;
	}

	public int getActualValueCount() {
		return actualValueCount;
	}

	public boolean compareCellValue (int r, int c, String value){
		if(isMatrix)
			try {
				return sampleMatrix.getValue(r, c) == UtilFunctions.objectToDouble(Types.ValueType.FP64, value);
			}
			catch(Exception exception){
				return false;
			}

		else
			try {
				return sampleFrame.get(r,c).equals(UtilFunctions.stringToObject(sampleFrame.getColumnType(c), value));
			}
			catch(Exception exception){
				return false;
			}
	}

	private class BuildRawIndexTask implements Callable<Object> {
		private final ArrayList<String> rawList;
		private RawIndex[] sampleRawIndex;
		private final int beginIndex;
		private final int endIndex;

		public BuildRawIndexTask(ArrayList<String> rawList, RawIndex[] sampleRawIndex, int beginIndex, int endIndex) {
			this.rawList = rawList;
			this.sampleRawIndex = sampleRawIndex;
			this.beginIndex = beginIndex;
			this.endIndex = endIndex;
		}

		@Override
		public Object call() throws Exception {
			for(int i=this.beginIndex; i<this.endIndex; i++)
				this.sampleRawIndex[i] = new RawIndex(this.rawList.get(i));
			return null;
		}
	}
}
