package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class CrossvalLop extends Lops {
		
	public CrossvalLop(Lops input) {
		super(Lops.Type.CrossvalLop, DataType.UNKNOWN, ValueType.UNKNOWN);
		
		
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
