/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class CMOperator extends Operator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
}
