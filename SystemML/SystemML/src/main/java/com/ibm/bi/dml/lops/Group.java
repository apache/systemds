/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;

/**
 * Lop to represent a grouping operation.
 */

public class Group extends Lop  
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public enum OperationTypes {Sort};
	
	OperationTypes operation;
	
	/**
	 * Constructor to create a grouping operation.
	 * @param input
	 * @param op
	 */

	public Group(Lop input, Group.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Grouping, dt, vt);		
		operation = op;
		this.addInput(input);
		input.addOutput(this);
		
		/*
		 *  This lop can be executed in only in GMR and RAND.
		 *  MMCJ, REBLOCK, and PARTITION themselves has location MapAndReduce.
		 */
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		
		boolean breaksAlignment = false;
		boolean aligner = true;
		boolean definesMRJob = true;
		
		this.lps.setProperties ( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() 
	{
		//return "Group " + "Operation: " + operation;
		return "Operation: " + operation;
	
	}

}
