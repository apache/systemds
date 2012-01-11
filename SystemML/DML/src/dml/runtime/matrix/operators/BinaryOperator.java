package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.Minus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.ValueFunction;

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
