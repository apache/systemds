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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class MatrixVectorBinaryOpFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 2814994742189295067L;
	
	private BinaryOperator _op;
	private Broadcast<PartitionedMatrixBlock> _pmV;
	private boolean _rightBroadcast;
	private boolean _isColumnVector; 
	private boolean _isOuter;
	
	public MatrixVectorBinaryOpFunction( BinaryOperator op, Broadcast<PartitionedMatrixBlock> binput, boolean rightBroadcast,
			boolean isColumnVector, boolean isOuter ) 
	{
		_op = op;
		_pmV = binput;
		_rightBroadcast = rightBroadcast;
		_isColumnVector = isColumnVector;
		_isOuter = isOuter;
	}

	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
		throws Exception 
	{
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		if( _isOuter ) 
		{
			int pNumRowBlocks = _pmV.value().getNumRowBlocks();
			int pNumColsBlocks = _pmV.value().getNumColumnBlocks();
			if(pNumColsBlocks == 1) {
				for(int i = 1; i <= pNumRowBlocks; i++ ) {
					MatrixBlock [] blocks = getBlocksForBinaryOperations(kv._2, i);
					MatrixBlock resultBlk = (MatrixBlock) (blocks[0].binaryOperations (_op, blocks[1], new MatrixBlock()));
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(i, kv._1.getColumnIndex()), resultBlk));
				}
			}
			else {
				for(int i = 1; i <= pNumColsBlocks; i++ ) {
					MatrixBlock [] blocks = getBlocksForBinaryOperations(kv._2, i);
					MatrixBlock resultBlk = (MatrixBlock) (blocks[0].binaryOperations (_op, blocks[1], new MatrixBlock()));
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(kv._1.getRowIndex(), i), resultBlk));
				}
			}
		}
		else {
			MatrixBlock [] blocks = null;
			if( _isColumnVector )
				blocks = getBlocksForBinaryOperations(kv._2, kv._1.getRowIndex());
			else
				blocks = getBlocksForBinaryOperations(kv._2, kv._1.getColumnIndex());
			
			MatrixBlock resultBlk = (MatrixBlock) (blocks[0].binaryOperations (_op, blocks[1], new MatrixBlock()));
			retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlk));
		}
		
		return retVal;
	}
	
	
	private MatrixBlock [] getBlocksForBinaryOperations(MatrixBlock streamingInputBlk, long index) throws DMLRuntimeException 
	{
		MatrixBlock blk1 = null;
		MatrixBlock blk2 = null;
		
		if( _rightBroadcast ) {
			blk1 = streamingInputBlk;
			if( _isColumnVector ) {
				blk2 = _pmV.value().getMatrixBlock((int) index,1);
			}
			else {
				blk2 = _pmV.value().getMatrixBlock(1, (int) index);
			}
		}
		else {
			blk2 = streamingInputBlk;
			if( _isColumnVector ) {
				blk2 = _pmV.value().getMatrixBlock((int) index,1);
			}
			else {
				blk2 = _pmV.value().getMatrixBlock(1, (int) index);
			}
		}
		
		MatrixBlock [] retVal = new MatrixBlock[2];
		retVal[0] = blk1;
		retVal[1] = blk2;
		return retVal;
	}
}
