package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SparseMatrix;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

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
