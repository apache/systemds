/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class MatrixMatrixBinaryOpFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -2683276102742977900L;
	
	private BinaryOperator _bop;
	
	public MatrixMatrixBinaryOpFunction(BinaryOperator op) {
		_bop = op;
	}

	@Override
	public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0)
			throws Exception 
	{
		return (MatrixBlock) arg0._1().binaryOperations(_bop, arg0._2(), new MatrixBlock());
	}
}