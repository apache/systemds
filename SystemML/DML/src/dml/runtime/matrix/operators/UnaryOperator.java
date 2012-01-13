package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.ValueFunction;

public class UnaryOperator  extends Operator {
	public ValueFunction fn;
	public UnaryOperator(ValueFunction p)
	{
		fn=p;
		if(fn instanceof Builtin)
		{
			Builtin f=(Builtin)fn;
			if(f.bFunc==Builtin.BuiltinFunctionCode.SIN || f.bFunc==Builtin.BuiltinFunctionCode.TAN)
				sparseSafe=true;
		}else
			sparseSafe=false;
	}
}
