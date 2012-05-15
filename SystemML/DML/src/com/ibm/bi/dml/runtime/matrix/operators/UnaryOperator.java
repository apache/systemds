package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class UnaryOperator  extends Operator {
	public ValueFunction fn;
	public UnaryOperator(ValueFunction p)
	{
		fn=p;
		if(fn instanceof Builtin)
		{
			Builtin f=(Builtin)fn;
			if(f.bFunc==Builtin.BuiltinFunctionCode.SIN || f.bFunc==Builtin.BuiltinFunctionCode.TAN || f.bFunc==Builtin.BuiltinFunctionCode.ROUND)
				sparseSafe=true;
		}else
			sparseSafe=false;
	}
}
