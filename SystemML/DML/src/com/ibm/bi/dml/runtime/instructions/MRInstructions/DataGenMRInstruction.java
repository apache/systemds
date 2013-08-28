package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.hops.Hops.DataGenMethod;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class DataGenMRInstruction extends MRInstruction {

	public byte input;
	public long rows;
	public long cols;
	public int rowsInBlock;
	public int colsInBlock;
	public String baseDir;
	private DataGenMethod method;
	
	public DataGenMRInstruction(Operator op, DataGenMethod mthd, byte in, byte out, long r, long c, int rpb, int cpb, String dir)
	{
		super(op, out);
		method = mthd;
		input=in;
		rows = r;
		cols = c;
		rowsInBlock = rpb;
		colsInBlock = cpb;
		baseDir = dir;
	}
	
	public DataGenMethod getDataGenMethod() {
		return method;
	}
	
	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input, output};
	}

}
