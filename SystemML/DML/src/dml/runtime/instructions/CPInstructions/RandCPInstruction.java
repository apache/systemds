package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;

public class RandCPInstruction extends UnaryCPInstruction{
	public long rows;
	public long cols;
	public double minValue;
	public double maxValue;
	public double sparsity;
	public String probabilityDensityFunction;
	public long seed=0;
	
	public RandCPInstruction (Operator op, 
							  CPOperand in, 
							  CPOperand out, 
							  long rows, 
							  long cols, 
							  double minValue, 
							  double maxValue,
							  double sparsity, 
							  String probabilityDensityFunction, 
							  String istr ) {
		super(op, in, out, istr);
		this.rows = rows;
		this.cols = cols;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.probabilityDensityFunction = probabilityDensityFunction;
	}

	public static Instruction parseInstruction(String str) 
	{
		Operator op = null;
		
		// Example: CP:Rand:rows=10:cols=10:min=0.0:max=1.0:sparsity=1.0:pdf=uniform:dir=scratch_space/_t0/:mVar0-MATRIX-DOUBLE
		String[] s = InstructionUtils.getInstructionPartsWithValueType(str);
		CPOperand in = null; // Rand instruction does not have any input matrices
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		out.split(s[s.length-1]); // ouput is specified by the last operand
		
		long rows = Long.parseLong(s[1].substring(5));
		long cols = Long.parseLong(s[2].substring(5));
		double minValue = Double.parseDouble(s[3].substring(4));
		double maxValue = Double.parseDouble(s[4].substring(4));
		double sparsity = Double.parseDouble(s[5].substring(9));
		String pdf = s[6].substring(4);
		
		return new RandCPInstruction(op, in, out, rows, cols, minValue, maxValue, sparsity, pdf, str);
	}
	
	public Data processInstruction (ProgramBlock pb)
		throws DMLRuntimeException{
		String output_name = output.get_name();
		
		MatrixObject sores = ((MatrixObject)pb.getVariable(output_name)).randOperations(rows, 
																				  cols, 
																				  minValue, 
																				  maxValue, 
																				  seed, 
																				  sparsity);
		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores;
	}
}
