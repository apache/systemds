/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;

/**
 * This aggregate function uses kahan+ with corrections to aggregate input blocks; it is meant for 
 * reducebykey operations where we CANNOT reuse the same correction block independent of the input
 * block indexes.
 * 
 */
public class AggregateSumMultiBlockFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -4015979658416853324L;

	private AggregateOperator _op = null;
	private MatrixBlock _corr = null;
	
	public AggregateSumMultiBlockFunction()
	{
		_op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);	
		_corr = new MatrixBlock();
	}
	
	@Override
	public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
		throws Exception 
	{
		//copy one input to output
		MatrixBlock out = new MatrixBlock(arg0);
		
		//aggregate other input
		_corr.reset(out.getNumRows(), out.getNumColumns());
		OperationsOnMatrixValues.incrementalAggregation(out, _corr, arg1, _op, false);
		
		return out;
	}
}
