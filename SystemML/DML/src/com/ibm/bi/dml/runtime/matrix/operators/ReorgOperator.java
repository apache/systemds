package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.IndexFunction;

public class ReorgOperator  extends Operator {
	public IndexFunction fn;
	
	public ReorgOperator(IndexFunction p)
	{
		fn=p;
		sparseSafe=true;
	}

}
