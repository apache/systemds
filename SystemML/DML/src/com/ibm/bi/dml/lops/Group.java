package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;

/**
 * Lop to represent a grouping operation.
 * @author aghoting
 */

public class Group extends Lops  
{
	public enum OperationTypes {Sort};
	
	OperationTypes operation;
	
	/**
	 * Constructor to create a grouping operation.
	 * @param input
	 * @param op
	 */

	public Group(Lops input, Group.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Grouping, dt, vt);		
		operation = op;
		this.addInput(input);
		input.addOutput(this);
		
		/*
		 *  This lop can be executed in only in GMR and RAND.
		 *  MMCJ, REBLOCK, and PARTITION themselves has location MapAndReduce.
		 */
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.RAND);
		
		boolean breaksAlignment = false;
		boolean aligner = true;
		boolean definesMRJob = true;
		
		this.lps.setProperties ( ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() 
	{
		//return "Group " + "Operation: " + operation;
		return "Operation: " + operation;
	
	}

}
