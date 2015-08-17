/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;


public class ReplicateVectorFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -1505557561471236851L;
	
	private boolean _byRow; 
	private long _numReplicas;
	
	public ReplicateVectorFunction(boolean byRow, long numReplicas) 
	{
		_byRow = byRow;
		_numReplicas = numReplicas;
	}
	
	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ix = arg0._1();
		MatrixBlock mb = arg0._2();
		
		//sanity check inputs
		if(_byRow && (ix.getRowIndex() != 1 || mb.getNumRows()>1) ) {
			throw new Exception("Expected a row vector in ReplicateVector");
		}
		if(!_byRow && (ix.getColumnIndex() != 1 || mb.getNumColumns()>1) ) {
			throw new Exception("Expected a column vector in ReplicateVector");
		}
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		for(int i = 1; i <= _numReplicas; i++) {
			if( _byRow )
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(i, ix.getColumnIndex()), mb));
			else
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(ix.getRowIndex(), i), mb));
		}
		
		return retVal;
	}
}
