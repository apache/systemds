/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.Iterator;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * 
 */
public class ResultMergeRemoteSparkWCompare extends ResultMerge implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -5970805069405942836L;
	
	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, MatrixBlock>> arg)
		throws Exception 
	{
		MatrixIndexes ixin = arg._1();
		Iterator<MatrixBlock> din = arg._2()._1().iterator();
		MatrixBlock cin = arg._2()._2();
		
		//create compare array
		double[][] compare = DataConverter.convertToDoubleMatrix(cin);
		
		//merge all blocks into compare block
		MatrixBlock out = new MatrixBlock(cin);
		while( din.hasNext() )
			mergeWithComp(out, din.next(), compare);
		
		//create output tuple
		return new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(ixin), out);
	}

	@Override
	public MatrixObject executeSerialMerge() 
			throws DMLRuntimeException 
	{
		throw new DMLRuntimeException("Unsupported operation.");
	}

	@Override
	public MatrixObject executeParallelMerge(int par)
			throws DMLRuntimeException 
	{
		throw new DMLRuntimeException("Unsupported operation.");
	}
}
