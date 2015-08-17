/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.lops.BinaryM.VectorType;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class MatrixVectorBinaryOpFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -7695883019452417300L;
	
	private BinaryOperator _op = null;
	private Broadcast<PartitionedMatrixBlock> _pmV = null;
	private VectorType _vtype = null;
	
	public MatrixVectorBinaryOpFunction( BinaryOperator op, Broadcast<PartitionedMatrixBlock> binput, VectorType vtype ) 
	{
		_op = op;
		_pmV = binput;
		_vtype = vtype;
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ix = arg0._1();
		MatrixBlock in1 = arg0._2();
		
		//get the rhs block 
		int rix= (int)((_vtype==VectorType.COL_VECTOR) ? ix.getRowIndex() : 1);
		int cix= (int)((_vtype==VectorType.COL_VECTOR) ? 1 : ix.getColumnIndex());
		MatrixBlock in2 = _pmV.value().getMatrixBlock(rix, cix);
			
		//execute the binary operation
		MatrixBlock ret = (MatrixBlock) (in1.binaryOperations (_op, in2, new MatrixBlock()));
		return new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(ix), ret);
	}
}
