/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.DiagIndex;
import com.ibm.bi.dml.runtime.functionobjects.IndexFunction;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

public class ReorgMapFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 31065772250744103L;
	
	private ReorgOperator _reorgOp = null;
	private IndexFunction _indexFnObject = null;
	
	public ReorgMapFunction(String opcode) 
			throws DMLRuntimeException 
	{
		if(opcode.equalsIgnoreCase("r'")) {
			_indexFnObject = SwapIndex.getSwapIndexFnObject();
		}
		else if(opcode.equalsIgnoreCase("rdiag")) {
			_indexFnObject = DiagIndex.getDiagIndexFnObject();
		}
		else {
			throw new DMLRuntimeException("Incorrect opcode for RDDReorgMapFunction:" + opcode);
		}
		_reorgOp = new ReorgOperator(_indexFnObject);
	}
	
	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
		throws Exception 
	{
		MatrixIndexes ixIn = arg0._1();
		MatrixBlock blkIn = arg0._2();

		//swap the matrix indexes
		MatrixIndexes ixOut = new MatrixIndexes(ixIn);
		_indexFnObject.execute(ixIn, ixOut);
		
		//swap the matrix block data
		MatrixBlock blkOut = (MatrixBlock) blkIn.reorgOperations(_reorgOp, new MatrixBlock(), -1, -1, -1);
		
		//output new tuple
		return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut,blkOut);
	}
	
}

