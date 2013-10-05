/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class RandInstruction extends DataGenMRInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public double minValue;
	public double maxValue;
	public double sparsity;
	public String probabilityDensityFunction;
	public long seed=0;
	
	public RandInstruction ( Operator op, byte in, byte out, long rows, long cols, int rpb, int cpb, double minValue, double maxValue,
				double sparsity, long seed, String probabilityDensityFunction, String baseDir, String istr ) {
		super(op, DataGenMethod.RAND, in, out, rows, cols, rpb, cpb, baseDir);
		mrtype = MRINSTRUCTION_TYPE.Rand;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.probabilityDensityFunction = probabilityDensityFunction;
		instString = istr;
	}
	
	public static Instruction parseInstruction(String str) throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 12 );

		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		Operator op = null;
		byte input = Byte.parseByte(parts[1]);
		byte output = Byte.parseByte(parts[2]);
		long rows = Double.valueOf(parts[3]).longValue();
		long cols = Double.valueOf(parts[4]).longValue();
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
