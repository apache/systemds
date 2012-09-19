package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


public class AggregateBinaryOperator extends Operator {

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
