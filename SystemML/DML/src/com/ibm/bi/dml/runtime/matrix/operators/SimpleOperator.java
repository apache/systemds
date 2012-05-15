package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.FunctionObject;

/*
 * Simple operator is just a wrapper for a single function object of any type.
 */
public class SimpleOperator extends Operator {
	public FunctionObject fn;
	
	public SimpleOperator ( FunctionObject f ) {
		fn = f;
	}
}
