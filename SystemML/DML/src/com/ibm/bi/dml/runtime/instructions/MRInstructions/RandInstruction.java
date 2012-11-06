package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
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
	
	public static Instruction parseInstruction(String str) throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 12 );

		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		Operator op = null;
		byte input = Byte.parseByte(parts[1]);
		byte output = Byte.parseByte(parts[2]);
		long rows = Long.parseLong(parts[3]);
		long cols = Long.parseLong(parts[4]);
		int rpb = Integer.parseInt(parts[5]);
		int cpb = Integer.parseInt(parts[6]);
		double minValue = Double.parseDouble(parts[7]);
		double maxValue = Double.parseDouble(parts[8]);
		double sparsity = Double.parseDouble(parts[9]);
		long seed = Long.parseLong(parts[10]);
		String pdf = parts[11];
		String baseDir = parts[12];
		
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
