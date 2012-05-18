package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.COV;

public class COVOperator extends Operator {

	public COV fn;
	public int constant;
	
	public COVOperator(COV op)
	{
		fn=op;
		sparseSafe=true; // TODO: check with YY
	}
}