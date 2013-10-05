/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.cost;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;

/**
 * 
 */
public class CostEstimatorNumMRJobs extends CostEstimator
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	protected double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args  ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		return 0;
	}
	
	@Override
	protected double getMRJobInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args  ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		return 1;
	}
}
