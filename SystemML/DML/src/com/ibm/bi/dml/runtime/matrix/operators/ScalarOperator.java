package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class ScalarOperator  extends Operator {
	public ValueFunction fn;
	public double constant;
	public ScalarOperator(ValueFunction p, double cst)
	{
		fn=p;
		constant=cst;
		//as long as (0 op v)=0, then op is sparsesafe
		if(fn instanceof Multiply || fn instanceof And)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	public void setConstant(double cst) {
		constant = cst;
	}
	public double executeScalar(double in) throws DMLRuntimeException {
		throw new DMLRuntimeException("executeScalar(): can not be invoked from base class.");
	}
}
