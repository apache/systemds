package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.MetaLearningFunctionParameters;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;

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
		this.lps.setProperties( ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
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
