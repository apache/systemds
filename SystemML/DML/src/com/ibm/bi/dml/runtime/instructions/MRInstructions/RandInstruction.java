package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class RandInstruction extends UnaryMRInstructionBase
{
	public long rows;
	public long cols;
	public int rowsInBlock;
	public int colsInBlock;
	public double minValue;
	public double maxValue;
	public double sparsity;
	public String probabilityDensityFunction;
	public String baseDir;
	public long seed=0;
	
	public RandInstruction ( Operator op, byte in, byte out, long rows, long cols, int rpb, int cpb, double minValue, double maxValue,
				double sparsity, long seed, String probabilityDensityFunction, String baseDir, String istr ) {
		super(op, in, out);
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
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
		int rpb = Integer.parseInt(s[6].substring(12));
		int cpb = Integer.parseInt(s[7].substring(12));
		double minValue = Double.parseDouble(s[8].substring(4));
		double maxValue = Double.parseDouble(s[9].substring(4));
		double sparsity = Double.parseDouble(s[10].substring(9));
		long seed = Long.parseLong(s[11].substring(5));
		String pdf = s[12].substring(4);
		String baseDir = s[13].substring(4);
		
		return new RandInstruction(op, input, output, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, baseDir, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		// TODO Auto-generated method stub
		
	}
	
}
