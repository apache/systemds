/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;


/**
 * Note: currently we always include the correction and use a subsequent maptopair to
 * drop them at the end because during aggregation we dont know if we produce an
 * intermediate or the final aggregate. 
 */
public class AggregateSingleBlockFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -3672377410407066396L;

	private AggregateOperator _op = null;
	private MatrixBlock _corr = null;
	
	public AggregateSingleBlockFunction( AggregateOperator op )
	{
		_op = op;	
		_corr = null;
	}
	
	@Override
	public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
		throws Exception 
	{
		//copy one first input
		MatrixBlock out = new MatrixBlock(arg0); 
		
		//create correction block (on demand)
		if( _corr == null ){
			_corr = new MatrixBlock(arg0.getNumRows(), arg0.getNumColumns(), false);
		}
		
		//aggregate second input
		if(_op.correctionExists) {
			OperationsOnMatrixValues.incrementalAggregation(
					out, _corr, arg1, _op, true);
		}
		else {
			OperationsOnMatrixValues.incrementalAggregation(
					out, null, arg1, _op, true);
		}
		
		return out;
	}
}
