/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

@Deprecated
public class ConvertALToBinaryBlockFunction implements PairFunction<Tuple2<MatrixIndexes, ArrayList<MatrixCell>>, MatrixIndexes, MatrixBlock> {
	
	
	private static final long serialVersionUID = -3672377410407066396L;
	long brlen; long bclen; 	// Block length: usually 1000 X 1000
	long rlen; long clen;		// Dimensionality of the matrix (number of rows: rlen and number of columns: clen)
	public ConvertALToBinaryBlockFunction(long brlen, long bclen, long rlen, long clen) {
		this.brlen = brlen;
		this.bclen = bclen;
		this.rlen = rlen;
		this.clen = clen;
	}
	
	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, ArrayList<MatrixCell>> kv) throws Exception {
		ArrayList<MatrixCell> cells = kv._2;
		// ------------------------------------------------------------------
		//	Compute local block size: 
		// Example: For matrix: 1500 X 1100 with block length 1000 X 1000
		// We will have four local block sizes (1000X1000, 1000X100, 500X1000 and 500X1000)
		long blockRowIndex = kv._1.getRowIndex();
		long blockColIndex = kv._1.getColumnIndex();
		int lrlen = UtilFunctions.computeBlockSize(rlen, blockRowIndex, brlen);
		int lclen = UtilFunctions.computeBlockSize(clen, blockColIndex, bclen);
		// ------------------------------------------------------------------
		
		// Create MatrixBlock
		int nnz = cells.size();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(lrlen, lclen, nnz);
		MatrixBlock mb = new MatrixBlock(lrlen, lclen, sparse, nnz);
		
		// copy values into new block
		for (MatrixCell cell : cells) {
			double value  =  cell.getValue();
			if ( value != 0 ) {
				if(cell.getRowIndex() < 0 || cell.getColIndex() < 0) {
					throw new Exception("Incorrect value for the cell:" + cell.toString());
				}
				// This function works for both sparse and dense blocks
				mb.appendValue(	(int)cell.getRowIndex(), 
						        (int)cell.getColIndex()
						          , value);
			}
		}
		// This function is a post-append step for sparse matrix block
		mb.sortSparseRows();
		
		return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, mb);
	}	
}
