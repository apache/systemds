package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.ValueFunction;


public class AggregateBinaryOperator extends Operator {

	public ValueFunction binaryFn;
	public AggregateOperator aggOp;
	
	public AggregateBinaryOperator(ValueFunction inner, AggregateOperator outer)
	{
		binaryFn=inner;
		aggOp=outer;
		if(binaryFn instanceof Multiply && aggOp.increOp.fn instanceof Plus)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
}
