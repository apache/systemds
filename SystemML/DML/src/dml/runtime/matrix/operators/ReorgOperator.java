package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.IndexFunction;

public class ReorgOperator  extends Operator {
	public IndexFunction fn;
	
	public ReorgOperator(IndexFunction p)
	{
		fn=p;
		sparseSafe=true;
	}

}
