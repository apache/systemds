package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


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
		//as long as (v op 0)=v, then op is sparseSafe
		if(op instanceof Plus || op instanceof Or || op instanceof Minus)
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
