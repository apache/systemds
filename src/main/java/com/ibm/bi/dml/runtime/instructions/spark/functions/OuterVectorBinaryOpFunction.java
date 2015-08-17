/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class OuterVectorBinaryOpFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1730704346934726826L;
	
	private BinaryOperator _op;
	private Broadcast<PartitionedMatrixBlock> _pmV;
	
	public OuterVectorBinaryOpFunction( BinaryOperator op, Broadcast<PartitionedMatrixBlock> binput ) 
	{
		_op = op;
		_pmV = binput;
	}

	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ix = arg0._1();
		MatrixBlock in1 = arg0._2();
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		int pNumColsBlocks = _pmV.value().getNumColumnBlocks();
		for(int i = 1; i <= pNumColsBlocks; i++ ) {
			MatrixBlock in2 = _pmV.value().getMatrixBlock(1, i);
			MatrixBlock resultBlk = (MatrixBlock) (in1.binaryOperations (_op, in2, new MatrixBlock()));
			retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(ix.getRowIndex(), i), resultBlk));
		}
		
		return retVal;
	}
}
