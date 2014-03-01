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
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.Expression.*;


public class PartitionLop extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	PartitionParams pp ;
	
	public PartitionLop(PartitionParams pp, Lop dataLop, DataType dt, ValueType vt) 
	{ 
		super(Lop.Type.PartitionLop, dt, vt);		
		this.pp = pp ;
		
		if(dataLop != null) 
		{
			this.addInput(dataLop) ;
			dataLop.addOutput(this) ;
		}
		
		/*
		 * This lop can be executed only in PARTITION job.
		 */
		boolean breaksAlignment = true; // TODO: verify
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.PARTITION);
		this.lps.setProperties(inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String getInstructions(int inputIndex, int outputIndex) throws LopsException 
	{
		throw new LopsException(this.printErrorLocation() + "Should never be invoked for partition lop");

	}

	@Override
	public String getInstructions(int inputIndex1, int inputIndex2,
			int outputIndex) throws LopsException 
	{
		throw new LopsException(this.printErrorLocation() + "Should never be invoked for partition lop");
	}

	@Override
	public String toString() {
		String s = "PartitionLop: " + pp.toString() ;
		return s ;
	}

	public PartitionParams getPartitionParams() {
		return pp;
	}
}