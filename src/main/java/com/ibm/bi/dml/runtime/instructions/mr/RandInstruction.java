/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class RandInstruction extends DataGenMRInstruction
{
	
	private double minValue;
	private double maxValue;
	private double sparsity;
	private String probabilityDensityFunction;
	private String pdfParams;
	
	private long seed=0;
	
	public RandInstruction ( Operator op, byte in, byte out, long rows, long cols, int rpb, int cpb, double minValue, double maxValue,
				double sparsity, long seed, String probabilityDensityFunction, String params, String baseDir, String istr ) {
		super(op, DataGenMethod.RAND, in, out, rows, cols, rpb, cpb, baseDir);
		mrtype = MRINSTRUCTION_TYPE.Rand;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.probabilityDensityFunction = probabilityDensityFunction;
		this.pdfParams = params;
		instString = istr;
	}
	
	public String getPdfParams() {
		return pdfParams;
	}
	
	public double getMinValue() {
		return minValue;
	}

	public double getMaxValue() {
		return maxValue;
	}

	public double getSparsity() {
		return sparsity;
	}

	public String getProbabilityDensityFunction() {
		return probabilityDensityFunction;
	}

	public long getSeed() {
		return seed;
	}

	public static Instruction parseInstruction(String str) throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 13 );

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
		String baseDir = parts[11];
		String pdf = parts[12];
		String pdfParams = parts[13];
		
		return new RandInstruction(op, input, output, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, pdfParams, baseDir, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		// TODO Auto-generated method stub
		
	}	
}
