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
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class CrossvalLop extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
			
	public CrossvalLop(Lop input) {
		super(Lop.Type.CrossvalLop, DataType.UNKNOWN, ValueType.UNKNOWN);
		
		
		if(input != null) {
			this.addInput(input) ;
			input.addOutput(this) ;
		}
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.PARTITION);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}
		
	@Override
	public String getInstructions(int inputIndex, int outputIndex) {
		return null;
	}

	@Override
	public String getInstructions(int inputIndex1, int inputIndex2,
			int outputIndex) {
		return null;
	}

	@Override
	public String toString() {
		String s = "crossval lop";
		return s ;
	}
	
}
