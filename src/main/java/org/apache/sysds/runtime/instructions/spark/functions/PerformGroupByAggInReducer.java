/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.WeightedCell;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class PerformGroupByAggInReducer implements Function<Iterable<WeightedCell>, WeightedCell> 
{
	private static final long serialVersionUID = 8160556441153227417L;
	
	Operator op;
	public PerformGroupByAggInReducer(Operator op) {
		this.op = op;
	}

	@Override
	public WeightedCell call(Iterable<WeightedCell> kv)
		throws Exception 
	{
		WeightedCell outCell = new WeightedCell();
		CmCovObject cmObj = new CmCovObject(); 
		if(op instanceof CMOperator) //everything except sum
		{
			cmObj.reset();
			CM lcmFn = CM.getCMFnObject(((CMOperator) op).aggOpType); // cmFn.get(key.getTag());
			if( ((CMOperator) op).isPartialAggregateOperator() ) {
				throw new DMLRuntimeException("Incorrect usage, should have used PerformGroupByAggInCombiner");
			}
			else //forward tuples to reducer
			{
				for(WeightedCell value : kv)
					lcmFn.execute(cmObj, value.getValue(), value.getWeight());
				
				outCell.setValue(cmObj.getRequiredResult(op));
				outCell.setWeight(1);
			}
		}
		else if(op instanceof AggregateOperator) //sum
		{
			AggregateOperator aggop=(AggregateOperator) op;
			
			if( aggop.existsCorrection() ) {
				KahanObject buffer=new KahanObject(aggop.initialValue, 0);
				KahanPlus.getKahanPlusFnObject();
				
				//partial aggregate with correction
				for(WeightedCell value : kv)
					aggop.increOp.fn.execute(buffer, value.getValue()*value.getWeight());
				
				outCell.setValue(buffer._sum);
				outCell.setWeight(1);
			}
			else { //no correction
				double v = aggop.initialValue;
				for(WeightedCell value : kv)
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
