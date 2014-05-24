/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * Factoring with minimum constraint (e.g., if communication is expensive)
 */
public class TaskPartitionerFactoringCmin extends TaskPartitionerFactoring
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected long _constraint = -1;
	
	public TaskPartitionerFactoringCmin( long taskSize, int numThreads, long constraint, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, numThreads, iterVarName, fromVal, toVal, incrVal);
		
		_constraint = constraint;
	}

	@Override
	protected long determineNextBatchSize(long R, int P) 
	{
		int x = 2;
		long K = (long) Math.ceil((double)R / ( x * P )); //NOTE: round creates more tasks
		
		if( K < _constraint ) //account for rounding errors
			K = _constraint;
		
		return K;
	}
	
}
