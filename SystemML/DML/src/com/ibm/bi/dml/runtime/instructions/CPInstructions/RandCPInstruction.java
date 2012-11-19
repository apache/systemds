package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.hops.RandOp;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class RandCPInstruction extends UnaryCPInstruction{
	public long rows;
	public long cols;
	public int rowsInBlock;
	public int colsInBlock;
	public double minValue;
	public double maxValue;
	public double sparsity;
	public String pdf;
	public long seed=0;
	
	public RandCPInstruction (Operator op, 
							  CPOperand in, 
							  CPOperand out, 
							  long rows, 
							  long cols,
							  int rpb, int cpb,
							  double minValue, 
							  double maxValue,
							  double sparsity, 
							  long seed,
							  String probabilityDensityFunction,
							  String istr) {
		super(op, in, out, istr);
		
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.pdf = probabilityDensityFunction;

	}

	public static Instruction parseInstruction(String str) throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 13 );
		
		Operator op = null;

		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		CPOperand in = null;
		
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		out.split(s[s.length-1]); // ouput is specified by the last operand
		
		long rows = -1, cols = -1;
        if (!s[3].contains( Lops.VARIABLE_NAME_PLACEHOLDER)) {
		   	rows = Long.parseLong(s[3]);
        }
        if (!s[4].contains( Lops.VARIABLE_NAME_PLACEHOLDER)) {
        	cols = Long.parseLong(s[4]);
        }
		
		int rpb = Integer.parseInt(s[5]);
		int cpb = Integer.parseInt(s[6]);
		double minValue = Double.parseDouble(s[7]);
		double maxValue = Double.parseDouble(s[8]);
		double sparsity = Double.parseDouble(s[9]);
		long seed = Long.parseLong(s[10]);
		String pdf = s[11];
		
		
		return new RandCPInstruction(op, in, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, str);
				
	}
	
	public void processInstruction (ProgramBlock pb)
		throws DMLRuntimeException
	{
		String output_name = output.get_name();
		
		//generate pseudo-random seed (because not specified) 
		long lSeed = seed; //seed per invocation
		if( lSeed == RandOp.UNSPECIFIED_SEED ) 
			lSeed = RandOp.generateRandomSeed();
		
		LOG.trace("Process RandCPInstruction with seed = "+lSeed+".");
		
		//execute rand
		MatrixBlock soresBlock = (MatrixBlock) (MatrixBlock.randOperations((int)rows, (int)cols, sparsity, minValue, maxValue, pdf, lSeed) );
        pb.setMatrixOutput(output_name, soresBlock);
	}
}
