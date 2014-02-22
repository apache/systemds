/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


public class AggregateBinaryOperator extends Operator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ValueFunction binaryFn;
	public AggregateOperator aggOp;
	
	public AggregateBinaryOperator(ValueFunction inner, AggregateOperator outer)
	{
		binaryFn=inner;
		aggOp=outer;
		//so far, we only support matrix multiplication, and it is sparseSafe
		if(binaryFn instanceof Multiply && aggOp.increOp.fn instanceof Plus)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
}
