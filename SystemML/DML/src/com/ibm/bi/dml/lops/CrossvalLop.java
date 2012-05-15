package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.MetaLearningFunctionParameters;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class CrossvalLop extends Lops {
		
	public MetaLearningFunctionParameters params ;
	
	public CrossvalLop(Lops input, MetaLearningFunctionParameters params) {
		super(Lops.Type.CrossvalLop, DataType.UNKNOWN, ValueType.UNKNOWN);
		
		//this.pp = pp ;
		this.params = params; 
		
		if(input != null) {
			this.addInput(input) ;
			input.addOutput(this) ;
		}
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.PARTITION);
		this.lps.setProperties( ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}
		
	public String getFunctionNames() {
		return "Tran: " + params.getTrainFunctionName() + " Test: " + params.getTestFunctionName();
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
		String s = getFunctionNames() ;
		return s ;
	}
	
}
