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

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SparseMatrix;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.SparseRowsIterator;
import org.apache.sysml.runtime.util.UtilFunctions;

public class GetMLLibBlocks implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, Tuple2<Object, Object>, Matrix> {

	private static final long serialVersionUID = 5977179886312384315L;
	
	long rlen; long clen; int brlen; int bclen;
	public GetMLLibBlocks(long rlen, long clen, int brlen, int bclen) {
		this.rlen = rlen;
		this.clen = clen;
		this.brlen = brlen;
		this.bclen = bclen;
	}
	
	private int getNNZ(MatrixBlock blk) {
		if(blk.getNonZeros() != -1) {
			return (int) blk.getNonZeros();
		}
		else if(blk.isInSparseFormat()) {
			SparseRowsIterator iter = blk.getSparseRowsIterator();
			int nnz = 0;
			while( iter.hasNext() ) {
				nnz++;
			}
			return nnz;
		}
		else {
			return blk.getDenseArray().length;
		}
	}
	
	private int [] getArray(ArrayList<Integer> al) {
		int [] retVal = new int[al.size()];
		int i = 0;
		for(Integer v : al) {
			retVal[i] = v;
			i++;
		}
		return retVal;
	}

	@Override
	public Tuple2<Tuple2<Object, Object>, Matrix> call(
			Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		Integer blockRowIndex = (int) kv._1.getRowIndex();
		Integer blockColIndex = (int) kv._1.getColumnIndex();
	
		Matrix mllibBlock = null;
		MatrixBlock blk = kv._2;
		// ------------------------------------------------------------------
		//	Compute local block size: 
		int lrlen = UtilFunctions.computeBlockSize(rlen, kv._1.getRowIndex(), brlen);
		int lclen = UtilFunctions.computeBlockSize(clen, kv._1.getColumnIndex(), bclen);
		// ------------------------------------------------------------------
				
		if(blk.isInSparseFormat()) {
			SparseRowsIterator iter = blk.getSparseRowsIterator();
			int nnz = getNNZ(blk);
			double [] values = new double[nnz];
			int [] rowIndices = new int[nnz];
			ArrayList<Integer> colPtrList = new ArrayList<Integer>();
			int lastCol = -1; int index = 0;
			while( iter.hasNext() ) {
				IJV cell = iter.next(); // TODO: This might output global
				// MLLib's sparse rows are stored as Compressed Sparse Column (CSC) format
				if(lastCol != cell.j) {
					lastCol = cell.j;
					colPtrList.add(lastCol);
				}
				try {
					values[index] = cell.v;
					rowIndices[index] = cell.i;
					index++;
				}
				catch(Exception e) {
					throw new Exception("The number of non-zeros are not set correctly.");
				}
			}
			
			int [] colPtrs = getArray(colPtrList);
			mllibBlock = new SparseMatrix(lrlen, lclen, colPtrs, rowIndices, values);
		}
		else {
			mllibBlock = new DenseMatrix(lrlen, lclen, blk.getDenseArray());
		}
		return new Tuple2<Tuple2<Object,Object>, Matrix>(new Tuple2<Object,Object>(blockRowIndex, blockColIndex), mllibBlock);
	}
	
}
