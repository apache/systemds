package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.ValueFunction;
import dml.utils.DMLRuntimeException;

public class RightScalarOperator extends ScalarOperator {

	public RightScalarOperator(ValueFunction p, double cst) {
		super(p, cst);
	}

	@Override
	public double executeScalar(double in) throws DMLRuntimeException {
		return fn.execute(in, constant);
	}
}
