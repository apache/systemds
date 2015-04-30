/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import java.io.Serializable;

import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


public class AggregateBinaryOperator extends Operator implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 1666421325090925726L;

	public ValueFunction binaryFn;
	public AggregateOperator aggOp;
	private int k; //num threads
	
	public AggregateBinaryOperator(ValueFunction inner, AggregateOperator outer)
	{
		//default degree of parallelism is 1 
		//(for example in MR/Spark because we parallelize over the number of blocks)
		this( inner, outer, 1 );
	}
	
	public AggregateBinaryOperator(ValueFunction inner, AggregateOperator outer, int numThreads)
	{
		binaryFn = inner;
		aggOp = outer;
		k = numThreads;
		
		//so far, we only support matrix multiplication, and it is sparseSafe
		if(binaryFn instanceof Multiply && aggOp.increOp.fn instanceof Plus)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
	public int getNumThreads() {
		return k;
	}
}
