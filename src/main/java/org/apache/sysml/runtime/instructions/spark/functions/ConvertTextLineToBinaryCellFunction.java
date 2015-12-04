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

package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.Converter;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.data.TextToBinaryCellConverter;
import org.apache.sysml.runtime.util.UtilFunctions;


public class ConvertTextLineToBinaryCellFunction implements PairFunction<String, MatrixIndexes, MatrixCell> {
	
	private static final long serialVersionUID = -3672377410407066396L;
	private int brlen; 
	private int bclen;
	private long rlen; 
	private long clen;
	
	public ConvertTextLineToBinaryCellFunction(MatrixCharacteristics mcOut) {
		this.brlen = mcOut.getRowsPerBlock();
		this.bclen = mcOut.getColsPerBlock();
		this.rlen = mcOut.getRows();
		this.clen = mcOut.getCols();
	}

	@SuppressWarnings("unchecked")
	@Override
	public Tuple2<MatrixIndexes, MatrixCell> call(String line) throws Exception {
		@SuppressWarnings("rawtypes")
		Converter converter = new TextToBinaryCellConverter();
		converter.setBlockSize(brlen, bclen);
		converter.convert(null, new Text(line));
		
		Pair<MatrixIndexes, MatrixValue> retVal = null;
		if(converter.hasNext()) {
			retVal = converter.next();
			
			if(retVal.getKey().getRowIndex() > rlen || retVal.getKey().getColumnIndex() > clen) {
				throw new Exception("Either incorrect metadata provided to text reblock (" + rlen + "," + clen
						+ ") or incorrect input line:" + line);
			}
			// ------------------------------------------------------------------------------------------
			// Get appropriate indexes for blockIndexes and cell
			// For line: 1020 704 2.362153706180234 (assuming default block size: 1000 X 1000),
			// blockRowIndex = 2, blockColIndex = 1, rowIndexInBlock = 19, colIndexInBlock = 703 ... TODO: double check this !!!
			long blockRowIndex = UtilFunctions.blockIndexCalculation(retVal.getKey().getRowIndex(), (int) brlen);
			long blockColIndex = UtilFunctions.blockIndexCalculation(retVal.getKey().getColumnIndex(), (int) bclen);
			long rowIndexInBlock = UtilFunctions.cellInBlockCalculation(retVal.getKey().getRowIndex(), brlen);
			long colIndexInBlock = UtilFunctions.cellInBlockCalculation(retVal.getKey().getColumnIndex(), bclen);
			// Perform sanity check
			if(blockRowIndex <= 0 || blockColIndex <= 0 || rowIndexInBlock < 0 || colIndexInBlock < 0) {
				throw new Exception("Error computing indexes for the line:" + line);
			}
			// ------------------------------------------------------------------------------------------
			
			MatrixIndexes blockIndexes = new MatrixIndexes(blockRowIndex, blockColIndex);
			MatrixCell cell = new MatrixCell(((MatrixCell)retVal.getValue()).getValue());
			
			return new Tuple2<MatrixIndexes, MatrixCell>(blockIndexes, cell);
		}
		
		// In case of header for matrix format
		return new Tuple2<MatrixIndexes, MatrixCell>(new MatrixIndexes(-1, -1), null);
	}
	
}

