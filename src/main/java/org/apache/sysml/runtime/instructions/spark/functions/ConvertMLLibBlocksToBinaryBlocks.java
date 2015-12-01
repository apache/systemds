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
package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SparseMatrix;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ConvertMLLibBlocksToBinaryBlocks implements PairFunction<Tuple2<Tuple2<Object,Object>,Matrix>, MatrixIndexes, MatrixBlock> {

	private static final long serialVersionUID = 1876492711549091662L;
	
	long rlen; long clen; int brlen; int bclen;
	public ConvertMLLibBlocksToBinaryBlocks(long rlen, long clen, int brlen, int bclen) {
		this.rlen = rlen;
		this.clen = clen;
		this.brlen = brlen;
		this.bclen = bclen;
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<Tuple2<Object, Object>, Matrix> kv) throws Exception {
		long nRows = (Long) kv._1._1;
		long nCols = (Long) kv._1._2;
		// ------------------------------------------------------------------
		//	Compute local block size: 
		int lrlen = UtilFunctions.computeBlockSize(rlen, nRows, brlen);
		int lclen = UtilFunctions.computeBlockSize(clen, nCols, bclen);
		// ------------------------------------------------------------------
		MatrixBlock blk = null;
		double [] vals = null;
		if(kv._2 instanceof DenseMatrix) {
			blk = new MatrixBlock(lrlen, lclen, false);
			vals = ((DenseMatrix) kv._2).values();
		}
		else if(kv._2 instanceof SparseMatrix) {
			blk = new MatrixBlock(lrlen, lclen, true);
			vals = ((SparseMatrix) kv._2).values();
		}
		else {
			throw new Exception("Unsupported type of matrix");
		}
		
		int iter = 0;
		for(int i = 0; i < lrlen-1; i++) {
			for(int j = 0; j < lclen-1; j++) {
				if(vals[iter] != 0)
					blk.addValue(i, j, vals[iter]);
				iter++;
			}
		}
		
		return new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(nRows, nCols), blk);
	}
	
}
