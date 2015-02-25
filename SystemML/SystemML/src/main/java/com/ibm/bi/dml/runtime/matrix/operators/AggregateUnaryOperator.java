/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.IndexFunction;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;


public class AggregateUnaryOperator  extends Operator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public AggregateOperator aggOp;
	public IndexFunction indexFn;
	
	public AggregateUnaryOperator(AggregateOperator aop, IndexFunction iop)
	{
		aggOp=aop;
		indexFn=iop;
		
		//decide on sparse safe
		if( aggOp.increOp.fn instanceof Plus || 
			aggOp.increOp.fn instanceof KahanPlus || 
			aggOp.increOp.fn instanceof Or || 
			aggOp.increOp.fn instanceof Minus ) 
		{
			sparseSafe=true;
		}
		else
			sparseSafe=false;
	}
}
