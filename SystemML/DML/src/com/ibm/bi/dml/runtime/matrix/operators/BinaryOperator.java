package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class BinaryOperator  extends Operator {
	public ValueFunction fn;
	public BinaryOperator(ValueFunction p)
	{
		fn=p;
		//as long as (0 op 0)=0, then op is sparseSafe
		if(fn instanceof Plus || fn instanceof Multiply || fn instanceof Minus 
				|| fn instanceof And || fn instanceof Or)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
}
