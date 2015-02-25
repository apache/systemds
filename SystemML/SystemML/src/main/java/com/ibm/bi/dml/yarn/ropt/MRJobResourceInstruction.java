/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;

/**
 * The purpose of this class is to encode the yarn mapred memory configuration into 
 * the generated runtime plan in order to take this information into account when
 * costing runtime plans. Having a subclass of MRJobInstructions allows for minimal 
 * interference with existing packages or costing in non-Yarn settings.
 * 
 */
public class MRJobResourceInstruction extends MRJobInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _maxMRTasks = -1;
	
	public MRJobResourceInstruction( MRJobInstruction that ) 
		throws IllegalArgumentException, IllegalAccessException
	{
		super( that );
	}
	
	public void setMaxMRTasks( long max )
	{
		_maxMRTasks = max;
	}
	
	public long getMaxMRTasks()
	{
		return _maxMRTasks;
	}

}
