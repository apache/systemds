/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;

/**
 * 
 */
public class MatrixScalarUnaryFunction implements Function<MatrixBlock,MatrixBlock> 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 7246757780621114500L;
	
	private ScalarOperator _op;
	
	public MatrixScalarUnaryFunction(ScalarOperator sc_op) {	
		_op = sc_op;
	}

	@Override
	public MatrixBlock call(MatrixBlock arg0) 
		throws Exception 
	{
		return (MatrixBlock) arg0.scalarOperations(_op, new MatrixBlock());
	}
}