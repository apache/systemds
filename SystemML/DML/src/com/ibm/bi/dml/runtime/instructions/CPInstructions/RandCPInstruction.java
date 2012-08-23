package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.RandOp;
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
							  double minValue, 
							  double maxValue,
							  double sparsity, 
							  long seed,
							  String probabilityDensityFunction, 
							  String istr ) {
		super(op, in, out, istr);
		this.rows = rows;
		this.cols = cols;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.pdf = probabilityDensityFunction;
	}

	public static Instruction parseInstruction(String str) 
	{
		Operator op = null;
		
		// Example: CP:Rand:rows=10:cols=10:min=0.0:max=1.0:sparsity=1.0:seed=7:pdf=uniform:dir=scratch_space/_t0/:mVar0-MATRIX-DOUBLE
		String[] s = InstructionUtils.getInstructionPartsWithValueType(str);
		CPOperand in = null; // Rand instruction does not have any input matrices
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		out.split(s[s.length-1]); // ouput is specified by the last operand
		
		long rows = Long.parseLong(s[1].substring(5));
		long cols = Long.parseLong(s[2].substring(5));
		double minValue = Double.parseDouble(s[3].substring(4));
		double maxValue = Double.parseDouble(s[4].substring(4));
		double sparsity = Double.parseDouble(s[5].substring(9));
		long seed = Long.parseLong(s[6].substring(5));
		String pdf = s[7].substring(4);
		
		return new RandCPInstruction(op, in, out, rows, cols, minValue, maxValue, sparsity, seed, pdf, str);
	}
	
	public void processInstruction (ProgramBlock pb)
		throws DMLRuntimeException
	{
		String output_name = output.get_name();
		
		//generate pseudo-random seed (because not specified) 
		long lSeed = seed; //seed per invocation
		if( lSeed == RandOp.UNSPECIFIED_SEED ) 
			lSeed = RandOp.generateRandomSeed();
		
		if( DMLScript.DEBUG )
			System.out.println("process RandCPInstruction with seed="+lSeed+".");
		
		//execute rand
		MatrixBlock soresBlock = (MatrixBlock) (MatrixBlock.randOperations((int)rows, (int)cols, sparsity, minValue, maxValue, pdf, lSeed) );
        pb.setMatrixOutput(output_name, soresBlock);
	}
}
