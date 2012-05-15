package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class RandInstruction extends UnaryMRInstructionBase
{
	public long rows;
	public long cols;
	public double minValue;
	public double maxValue;
	public double sparsity;
	public String probabilityDensityFunction;
	public String baseDir;
	public long seed=0;
	
	public RandInstruction ( Operator op, byte in, byte out, long rows, long cols, double minValue, double maxValue,
				double sparsity, String probabilityDensityFunction, String baseDir, String istr ) {
		super(op, in, out);
		this.rows = rows;
		this.cols = cols;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.probabilityDensityFunction = probabilityDensityFunction;
		this.baseDir = baseDir;
		instString = istr;
	}
	
	public static Instruction parseInstruction(String str) 
	{
		String[] s = str.split(Instruction.OPERAND_DELIM);
		
		// s[0]: CP or MR (exec type)
		// s[1]: Rand (opcode)
		
		Operator op = null;
		byte input = Byte.parseByte(s[2]);
		byte output = Byte.parseByte(s[3]);
		long rows = Long.parseLong(s[4].substring(5));
		long cols = Long.parseLong(s[5].substring(5));
		double minValue = Double.parseDouble(s[6].substring(4));
		double maxValue = Double.parseDouble(s[7].substring(4));
		double sparsity = Double.parseDouble(s[8].substring(9));
		String pdf = s[9].substring(4);
		String baseDir = s[10].substring(4);
		
		return new RandInstruction(op, input, output, rows, cols, minValue, maxValue, sparsity, pdf, baseDir, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		// TODO Auto-generated method stub
		
	}
	
}
