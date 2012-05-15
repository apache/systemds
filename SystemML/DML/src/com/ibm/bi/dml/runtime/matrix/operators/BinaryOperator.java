package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class BinaryOperator  extends Operator {
	public ValueFunction fn;
	public BinaryOperator(ValueFunction p)
	{
		fn=p;
		if(fn instanceof Plus || fn instanceof Multiply || fn instanceof Minus)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
}
