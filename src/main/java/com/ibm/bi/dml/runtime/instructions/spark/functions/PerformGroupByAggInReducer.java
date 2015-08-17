/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.cp.KahanObject;
import com.ibm.bi.dml.runtime.matrix.data.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class PerformGroupByAggInReducer implements PairFunction<Tuple2<Long,Iterable<WeightedCell>>, Long, WeightedCell> {

	private static final long serialVersionUID = 8160556441153227417L;
	
	Operator op;
	public PerformGroupByAggInReducer(Operator op) {
		this.op = op;
	}

	@Override
	public Tuple2<Long, WeightedCell> call(
			Tuple2<Long, Iterable<WeightedCell>> kv) throws Exception {
		return new Tuple2<Long, WeightedCell>(kv._1, doAggregation(op, kv._2));
	}
	
	public WeightedCell doAggregation(Operator op, Iterable<WeightedCell> values) throws DMLRuntimeException {
		WeightedCell outCell = new WeightedCell();
		CM_COV_Object cmObj = new CM_COV_Object(); 
		if(op instanceof CMOperator) //everything except sum
		{
			cmObj.reset();
			CM lcmFn = CM.getCMFnObject(((CMOperator) op).aggOpType); // cmFn.get(key.getTag());
			if( ((CMOperator) op).isPartialAggregateOperator() )
			{
				throw new DMLRuntimeException("Incorrect usage, should have used PerformGroupByAggInCombiner");
				
//				//partial aggregate cm operator
//				for(WeightedCell value : values)
//					lcmFn.execute(cmObj, value.getValue(), value.getWeight());
//				
//				outCell.setValue(cmObj.getRequiredPartialResult(op));
//				outCell.setWeight(cmObj.getWeight());
			}
			else //forward tuples to reducer
			{
				for(WeightedCell value : values)
					lcmFn.execute(cmObj, value.getValue(), value.getWeight());
				
				outCell.setValue(cmObj.getRequiredResult(op));
				outCell.setWeight(1);
			}
		}
		else if(op instanceof AggregateOperator) //sum
		{
			AggregateOperator aggop=(AggregateOperator) op;
				
			if( aggop.correctionExists ) {
				KahanObject buffer=new KahanObject(aggop.initialValue, 0);
				
				KahanPlus.getKahanPlusFnObject();
				
				//partial aggregate with correction
				for(WeightedCell value : values)
					aggop.increOp.fn.execute(buffer, value.getValue()*value.getWeight());
				
				outCell.setValue(buffer._sum);
				outCell.setWeight(1);
			}
			else //no correction
			{
				double v = aggop.initialValue;
				
				//partial aggregate without correction
				for(WeightedCell value : values)
					v=aggop.increOp.fn.execute(v, value.getValue()*value.getWeight());
				
				outCell.setValue(v);
				outCell.setWeight(1);
			}				
		}
		else
			throw new DMLRuntimeException("Unsupported operator in grouped aggregate instruction:" + op);
		
		return outCell;
	}
	
}
