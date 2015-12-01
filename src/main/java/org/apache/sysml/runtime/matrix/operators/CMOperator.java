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


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class CMOperator extends Operator 
{
	
	private static final long serialVersionUID = 4126894676505115420L;
	
	// supported aggregates
	public enum AggregateOperationTypes {
		SUM, COUNT, MEAN, CM, CM2, CM3, CM4, NORM, VARIANCE, INVALID
	};

	public ValueFunction fn;
	public AggregateOperationTypes aggOpType;

	public CMOperator(ValueFunction op, AggregateOperationTypes agg) {
		fn = op;
		aggOpType = agg;
		sparseSafe = true;
	}

	public AggregateOperationTypes getAggOpType() {
		return aggOpType;
	}
	
	public void setCMAggOp(int order) {
		aggOpType = getCMAggOpType(order);
		fn = CM.getCMFnObject(aggOpType);
	}
	
	public static AggregateOperationTypes getCMAggOpType ( int order ) {
		if ( order == 2 )
			return AggregateOperationTypes.CM2;
		else if ( order == 3 )
			return AggregateOperationTypes.CM3;
		else if ( order == 4 )
			return AggregateOperationTypes.CM4;
		else if( order == 0)//this is a special case to handel weighted mean
			return AggregateOperationTypes.MEAN;
		else 
			return AggregateOperationTypes.INVALID;
	}
	
	public static AggregateOperationTypes getAggOpType(String fn, String order) {
		if (fn.equalsIgnoreCase("count")) {
			return AggregateOperationTypes.COUNT;
		} else if (fn.equalsIgnoreCase("sum")) {
			return AggregateOperationTypes.SUM;
		} else if (fn.equalsIgnoreCase("mean")) {
			return AggregateOperationTypes.MEAN;
		} else if (fn.equalsIgnoreCase("variance")) {
			return AggregateOperationTypes.VARIANCE;
		} else if (fn.equalsIgnoreCase("centralmoment")) {
			// in case of centralmoment, find aggIo by order
			if ( order == null )
				return AggregateOperationTypes.INVALID;
			
			if (order.equalsIgnoreCase("2"))
				return AggregateOperationTypes.CM2;
			else if (order.equalsIgnoreCase("3"))
				return AggregateOperationTypes.CM3;
			else if (order.equalsIgnoreCase("4"))
				return AggregateOperationTypes.CM4;
			else
				return AggregateOperationTypes.INVALID;
		}
		return AggregateOperationTypes.INVALID;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isPartialAggregateOperator()
	{
		boolean ret = false;
	
		switch( aggOpType )
		{
			case COUNT:
			case MEAN: 
				ret = true; break;
				
			//NOTE: the following aggregation operators are not marked for partial aggregation 
			//because they required multiple intermediate values and hence do not apply to the 
			//grouped aggregate combiner which needs to work on value/weight pairs only.
			case CM2:
			case CM3:
			case CM4:
			case VARIANCE: 
				ret = false; break;	
			
			default:
				//do nothing
		}
		
		return ret;
	}
}
