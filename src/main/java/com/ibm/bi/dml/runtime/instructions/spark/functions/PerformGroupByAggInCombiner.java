package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.cp.KahanObject;
import com.ibm.bi.dml.runtime.matrix.data.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class PerformGroupByAggInCombiner implements Function2<WeightedCell, WeightedCell, WeightedCell> {

	private static final long serialVersionUID = -813530414567786509L;
	
	Operator op;
	public PerformGroupByAggInCombiner(Operator op) {
		this.op = op;
	}

	@Override
	public WeightedCell call(WeightedCell value1, WeightedCell value2) throws Exception {
		return doAggregation(op, value1, value2);
	}

	public WeightedCell doAggregation(Operator op, WeightedCell value1, WeightedCell value2) throws DMLRuntimeException {
		WeightedCell outCell = new WeightedCell();
		CM_COV_Object cmObj = new CM_COV_Object(); 
		if(op instanceof CMOperator) //everything except sum
		{
			if( ((CMOperator) op).isPartialAggregateOperator() )
			{
				cmObj.reset();
				CM lcmFn = CM.getCMFnObject(((CMOperator) op).aggOpType); // cmFn.get(key.getTag());
				
				//partial aggregate cm operator
				lcmFn.execute(cmObj, value1.getValue(), value1.getWeight());
				lcmFn.execute(cmObj, value2.getValue(), value2.getWeight());
				
				outCell.setValue(cmObj.getRequiredPartialResult(op));
				outCell.setWeight(cmObj.getWeight());	
			}
			else //forward tuples to reducer
			{
				throw new DMLRuntimeException("Incorrect usage, should have used PerformGroupByAggInReducer");
			}				
		}
		else if(op instanceof AggregateOperator) //sum
		{
			AggregateOperator aggop=(AggregateOperator) op;
				
			if( aggop.correctionExists ) {
				KahanObject buffer=new KahanObject(aggop.initialValue, 0);
				
				KahanPlus.getKahanPlusFnObject();
				
				//partial aggregate with correction
				aggop.increOp.fn.execute(buffer, value1.getValue()*value1.getWeight());
				aggop.increOp.fn.execute(buffer, value2.getValue()*value2.getWeight());
				
				outCell.setValue(buffer._sum);
				outCell.setWeight(1);
			}
			else //no correction
			{
				double v = aggop.initialValue;
				
				//partial aggregate without correction
				v=aggop.increOp.fn.execute(v, value1.getValue()*value1.getWeight());
				v=aggop.increOp.fn.execute(v, value2.getValue()*value2.getWeight());
				
				outCell.setValue(v);
				outCell.setWeight(1);
			}				
		}
		else
			throw new DMLRuntimeException("Unsupported operator in grouped aggregate instruction:" + op);
		
		return outCell;
	}
}
