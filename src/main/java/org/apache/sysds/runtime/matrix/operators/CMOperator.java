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


package org.apache.sysds.runtime.matrix.operators;

import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.ValueFunction;

public class CMOperator extends MultiThreadedOperator
{
	private static final long serialVersionUID = 4126894676505115420L;
	
	// supported aggregates
	public enum AggregateOperationTypes {
		SUM,
		COUNT,
		MEAN, //a.k.a. CM
		CM2,
		CM3,
		CM4,
		MIN,
		MAX,
		VARIANCE,
		INVALID
	}

	public final ValueFunction fn;
	public final AggregateOperationTypes aggOpType;

	public CMOperator(ValueFunction op, AggregateOperationTypes agg) {
		this(op, agg, 1);
	}
	
	public CMOperator(ValueFunction op, AggregateOperationTypes agg, int numThreads) {
		super(true);
		fn = op;
		aggOpType = agg;
		_numThreads = numThreads;
	}

	public CMOperator(CMOperator that) {
		// Deep copy the stateful ValueFunction
		fn = that.fn instanceof CM ? CM.getCMFnObject((CM)that.fn) : that.fn;
		aggOpType = that.aggOpType;
		_numThreads = that._numThreads;
	}

	public AggregateOperationTypes getAggOpType() {
		return aggOpType;
	}

	public CMOperator setCMAggOp(int order) {
		AggregateOperationTypes agg = getCMAggOpType(order);
		ValueFunction fn = CM.getCMFnObject(aggOpType);
		return new CMOperator(fn, agg, _numThreads);
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
		} else if (fn.equalsIgnoreCase("min")) {
			return AggregateOperationTypes.MIN;
		} else if (fn.equalsIgnoreCase("max")) {
			return AggregateOperationTypes.MAX;
		}
		return AggregateOperationTypes.INVALID;
	}

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
