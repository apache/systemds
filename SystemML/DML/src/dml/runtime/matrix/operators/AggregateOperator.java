package dml.runtime.matrix.operators;

import dml.runtime.functionobjects.Minus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.ValueFunction;

public class AggregateOperator  extends Operator {

	public double initialValue;
	public BinaryOperator increOp;
	
	public boolean correctionExists=false;
	public byte correctionLocation=0;//0 means none, 1 means the last row, 2 means the last column, 
									//3 means last row is correction 2nd last is count
	                                //4 means last column is correction 2nd last is count
	
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
	
	public AggregateOperator(double initValue, ValueFunction op, boolean correctionExists, byte correctionLocation)
	{
		this(initValue, op);
		this.correctionExists=correctionExists;
		this.correctionLocation=correctionLocation;
	}
	
}
