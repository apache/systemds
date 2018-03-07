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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.DataGen;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;

public class DataGenCPInstruction extends UnaryCPInstruction {

	private DataGenMethod method = DataGenMethod.INVALID;

	private final CPOperand rows, cols;
	private final int rowsInBlock, colsInBlock;
	private final double minValue, maxValue, sparsity;
	private final String pdf, pdfParams;
	private final long seed;

	// sequence specific attributes
	private final CPOperand seq_from, seq_to, seq_incr;

	// sample specific attributes
	private final boolean replace;
	private final int numThreads;

	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, 
			CPOperand rows, CPOperand cols, int rpb, int cpb, double minValue, double maxValue, double sparsity, long seed,
			String probabilityDensityFunction, String pdfParams, int k, 
			CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, boolean replace, String opcode, String istr) {
		super(CPType.Rand, op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.pdf = probabilityDensityFunction;
		this.pdfParams = pdfParams;
		this.numThreads = k;
		this.seq_from = seqFrom;
		this.seq_to = seqTo;
		this.seq_incr = seqIncr;
		this.replace = replace;
	}
	
	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, double minValue, double maxValue, double sparsity, long seed,
			String probabilityDensityFunction, String pdfParams, int k, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, 
			probabilityDensityFunction, pdfParams, k, null, null, null, false, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, double maxValue, boolean replace, long seed, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, rpb, cpb, 0, maxValue, 1.0, seed, 
			null, null, 1, null, null, null, replace, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, rpb, cpb, 0, 1, 1.0, -1, 
			null, null, 1, seqFrom, seqTo, seqIncr, false, opcode, istr);
	}

	public long getRows() {
		return rows.isLiteral() ? Long.parseLong(rows.getName()) : -1;
	}

	public long getCols() {
		return cols.isLiteral() ? Long.parseLong(cols.getName()) : -1;
	}
	
	public int getRowsInBlock() {
		return rowsInBlock;
	}

	public int getColsInBlock() {
		return colsInBlock;
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

	public static DataGenCPInstruction parseInstruction(String str) 
		throws DMLRuntimeException 
	{
		DataGenMethod method = DataGenMethod.INVALID;

		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = s[0];
		
		if ( opcode.equalsIgnoreCase(DataGen.RAND_OPCODE) ) {
			method = DataGenMethod.RAND;
			InstructionUtils.checkNumFields ( s, 12 );
		}
		else if ( opcode.equalsIgnoreCase(DataGen.SEQ_OPCODE) ) {
			method = DataGenMethod.SEQ;
			// 8 operands: rows, cols, rpb, cpb, from, to, incr, outvar
			InstructionUtils.checkNumFields ( s, 8 ); 
		}
		else if ( opcode.equalsIgnoreCase(DataGen.SAMPLE_OPCODE) ) {
			method = DataGenMethod.SAMPLE;
			// 7 operands: range, size, replace, seed, rpb, cpb, outvar
			InstructionUtils.checkNumFields ( s, 7 ); 
		}
		
		CPOperand out = new CPOperand(s[s.length-1]);
		Operator op = null;
		
		if ( method == DataGenMethod.RAND ) 
		{
			CPOperand rows = new CPOperand(s[1]);
			CPOperand cols = new CPOperand(s[2]);
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			double minValue = !s[5].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[5]).doubleValue() : -1;
			double maxValue = !s[6].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[6]).doubleValue() : -1;
			double sparsity = !s[7].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[7]).doubleValue() : -1;
			long seed = !s[8].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Long.valueOf(s[8]).longValue() : -1;
			String pdf = s[9];
			String pdfParams = !s[10].contains( Lop.VARIABLE_NAME_PLACEHOLDER) ?
				s[10] : null;
			int k = Integer.parseInt(s[11]);
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, pdfParams, k, opcode, str);
		}
		else if ( method == DataGenMethod.SEQ) 
		{
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			CPOperand from = new CPOperand(s[5]);
			CPOperand to = new CPOperand(s[6]);
			CPOperand incr = new CPOperand(s[7]);
			
			return new DataGenCPInstruction(op, method, null, out, null, null, rpb, cpb, from, to, incr, opcode, str);
		}
		else if ( method == DataGenMethod.SAMPLE) 
		{
			double max = !s[1].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[1]) : 0;
			CPOperand rows = new CPOperand(s[2]);
			CPOperand cols = new CPOperand("1", ValueType.INT, DataType.SCALAR);
			boolean replace = (!s[3].contains(Lop.VARIABLE_NAME_PLACEHOLDER) 
				&& Boolean.valueOf(s[3]));
			
			long seed = Long.parseLong(s[4]);
			int rpb = Integer.parseInt(s[5]);
			int cpb = Integer.parseInt(s[6]);
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, rpb, cpb, max, replace, seed, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		MatrixBlock soresBlock = null;
		
		//process specific datagen operator
		if ( method == DataGenMethod.RAND ) {
			long lrows = ec.getScalarInput(rows).getLongValue();
			long lcols = ec.getScalarInput(cols).getLongValue();
			checkValidDimensions(lrows, lcols);
			
			//generate pseudo-random seed (because not specified) 
			long lSeed = seed; //seed per invocation
			if( lSeed == DataGenOp.UNSPECIFIED_SEED ) 
				lSeed = DataGenOp.generateRandomSeed();
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction rand with seed = "+lSeed+".");
			
			RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
				pdf, (int) lrows, (int) lcols, rowsInBlock, colsInBlock, sparsity, minValue, maxValue, pdfParams);
			soresBlock = MatrixBlock.randOperations(rgen, seed, numThreads);
		}
		else if ( method == DataGenMethod.SEQ ) 
		{
			double lfrom = ec.getScalarInput(seq_from).getDoubleValue();
			double lto = ec.getScalarInput(seq_to).getDoubleValue();
			double lincr = ec.getScalarInput(seq_incr).getDoubleValue();
			
			//handle default 1 to -1 for special case of from>to
			lincr = LibMatrixDatagen.updateSeqIncr(lfrom, lto, lincr);
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction seq with seqFrom="+lfrom+", seqTo="+lto+", seqIncr"+lincr);
			
			soresBlock = MatrixBlock.seqOperations(lfrom, lto, lincr);
		}
		else if ( method == DataGenMethod.SAMPLE ) 
		{
			long lrows = ec.getScalarInput(rows).getLongValue();
			long range = UtilFunctions.toLong(maxValue);
			checkValidDimensions(lrows, 1);
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction sample with range="+range+", size="+lrows+", replace"+replace + ", seed=" + seed);
			
			if ( range < lrows && !replace )
				throw new DMLRuntimeException("Sample (size=" + lrows + ") larger than population (size=" + range + ") can only be generated with replacement.");
			
			soresBlock = MatrixBlock.sampleOperations(range, (int)lrows, replace, seed);
		}
		
		//guarded sparse block representation change
		if( soresBlock.getInMemorySize() < OptimizerUtils.SAFE_REP_CHANGE_THRES )
			soresBlock.examSparsity();
		
		//release created output
		ec.setMatrixOutput(output.getName(), soresBlock, getExtendedOpcode());
	}
	
	private static void checkValidDimensions(long rows, long cols) throws DMLRuntimeException {
		//check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if( rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE )
			throw new DMLRuntimeException("DataGenCPInstruction does not "
				+ "support dimensions larger than integer: rows="+rows+", cols="+cols+".");
	}
}
