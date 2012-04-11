package dml.runtime.matrix.operators;

import dml.lops.PartialAggregate.CorrectionLocationType;
import dml.runtime.functionobjects.Minus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.ValueFunction;

public class AggregateOperator  extends Operator {

	public double initialValue;
	public BinaryOperator increOp;
	
	public boolean correctionExists=false;
	public CorrectionLocationType correctionLocation=CorrectionLocationType.INVALID;
	
	public AggregateOperator(double initValue, ValueFunction op)
	{
		initialValue=initValue;
		increOp=new BinaryOperator(op);
		//increFn=op;
		if(op instanceof Plus || op instanceof Multiply || op instanceof Minus)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
	public AggregateOperator(double initValue, ValueFunction op, boolean correctionExists, CorrectionLocationType correctionLocation)
	{
		this(initValue, op);
		this.correctionExists=correctionExists;
		this.correctionLocation=correctionLocation;
	}
	
}
