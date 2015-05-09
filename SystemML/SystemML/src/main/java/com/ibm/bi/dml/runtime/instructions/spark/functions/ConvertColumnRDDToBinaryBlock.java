package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class ConvertColumnRDDToBinaryBlock {

	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryBlockedRDD(JavaRDD<Double> columnRDD, int brlen, long rlen, SparkExecutionContext sec) {
		
		// Since getRowOffsets requires a collect to get linecount, might as well do a collect here and convert it to RDD as CP operation.
		// This limitation also exists for CSVReblock
		// TODO: Optimize this for large column vectors !!!
		
		List<Double> values = columnRDD.collect();
		
		int i = 0; int brIndex = 1;
		// ------------------------------------------------------------------
		//	Compute local block size: 
		int lrlen = UtilFunctions.computeBlockSize(rlen, brIndex, brlen);
		// ------------------------------------------------------------------
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		MatrixBlock blk = new MatrixBlock(lrlen, 1, true);
		for(Double value : values) {
			if(i == lrlen) {
				// Block filled in. Create new block.
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(brIndex, 1), blk));
				brIndex++;
				lrlen = UtilFunctions.computeBlockSize(rlen, brIndex, brlen);
				blk = new MatrixBlock(lrlen, 1, true);
				i = 0;
			}
			blk.appendValue(i, 0, value);
			// blk.setValue(i, 1, value);
			i++;
		}
		
		// Now insert last block
		retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(brIndex, 1), blk));
		
		return JavaPairRDD.fromJavaRDD(sec.getSparkContext().parallelize(retVal));
	}
}
