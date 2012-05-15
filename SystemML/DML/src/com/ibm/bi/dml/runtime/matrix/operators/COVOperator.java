package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class COVOperator extends Operator {

	public ValueFunction increOp;
	public int constant;
	
	public COVOperator(ValueFunction op)
	{
		increOp=op;
		sparseSafe=true;
	}
}