package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.Row;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class GetMIMBFromRow implements PairFunction<Row, MatrixIndexes, MatrixBlock> {

	private static final long serialVersionUID = 2291741087248847581L;

	@Override	
	public Tuple2<MatrixIndexes, MatrixBlock> call(Row row) throws Exception {
//		try {
			MatrixIndexes indx = (MatrixIndexes) row.apply(0);
			MatrixBlock blk = (MatrixBlock) row.apply(1);
			return new Tuple2<MatrixIndexes, MatrixBlock>(indx, blk);
//		}
//		catch(Exception e) {
//			throw new Exception("Incorrect type of DataFrame passed to MLMatrix:" + e.getMessage());
//		}
	}

}
