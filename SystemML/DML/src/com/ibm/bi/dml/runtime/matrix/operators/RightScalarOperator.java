package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class RightScalarOperator extends ScalarOperator {

	public RightScalarOperator(ValueFunction p, double cst) {
		super(p, cst);
	}

	@Override
	public double executeScalar(double in) throws DMLRuntimeException {
		return fn.execute(in, constant);
	}
}
