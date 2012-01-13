package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.ValueFunction;
import dml.utils.DMLRuntimeException;

public class ScalarOperator  extends Operator {
	public ValueFunction fn;
	public double constant;
	public ScalarOperator(ValueFunction p, double cst)
	{
		fn=p;
		constant=cst;
		if(fn instanceof Multiply)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
	public double executeScalar(double in) throws DMLRuntimeException {
		throw new DMLRuntimeException("executeScalar(): can not be invoked from base class.");
	}
}
