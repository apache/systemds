/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.instructions.mr;

import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;


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

	public static RandInstruction parseInstruction(String str) throws DMLRuntimeException 
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
			throws DMLRuntimeException 
	{
		//do nothing (only meta carrier, handled in special job type)
	}	
}
