package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;

import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

import scala.Tuple2;

public class CopyBinaryCellFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixCell>,MatrixIndexes, MatrixCell> 
{

	private static final long serialVersionUID = -675490732439827014L;

	public CopyBinaryCellFunction(  ) {
	
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixCell> call(
		Tuple2<MatrixIndexes, MatrixCell> arg0) throws Exception {
		MatrixIndexes ix = new MatrixIndexes(arg0._1());
		MatrixCell cell = new MatrixCell();
		cell.copy(arg0._2());
		return new Tuple2<MatrixIndexes,MatrixCell>(ix,cell);
	}
}