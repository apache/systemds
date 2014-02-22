/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * Factoring with maximum constraint (e.g., if LIX matrix out-of-core and we need
 * to bound the maximum number of iterations per map task -> memory bounds) 
 */
public class TaskPartitionerFactoringCmax extends TaskPartitionerFactoring
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected int _constraint = -1;
	
	public TaskPartitionerFactoringCmax( int taskSize, int numThreads, int constraint, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, numThreads, iterVarName, fromVal, toVal, incrVal);
		
		_constraint = constraint;
	}

	@Override
	protected int determineNextBatchSize(int R, int P) 
	{
		int x = 2;
		int K = (int) Math.ceil((double)R / ( x * P )); //NOTE: round creates more tasks
		
		if( K > _constraint ) //account for rounding errors
			K = _constraint;
		
		return K;
	}
	
}
