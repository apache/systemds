/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.hops.DataGenOp;
import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.DataGen;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.data.BasicTensor;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.lineage.LineageItem;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public class DataGenCPInstruction extends UnaryCPInstruction {

	private DataGenMethod method;

	private final CPOperand rows, cols, dims;
	private final int rowsInBlock, colsInBlock;
	private boolean minMaxAreDoubles;
	private final String minValueStr, maxValueStr;
	private final double minValue, maxValue, sparsity;
	private final String pdf, pdfParams;
	private final long seed;
	private Long runtimeSeed;

	// sequence specific attributes
	private final CPOperand seq_from, seq_to, seq_incr;

	// sample specific attributes
	private final boolean replace;
	private final int numThreads;

	// seed positions
	private static final int SEED_POSITION_RAND = 9;
	private static final int SEED_POSITION_SAMPLE = 4;

	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, 
			CPOperand rows, CPOperand cols, CPOperand dims, int rpb, int cpb, String minValue, String maxValue, double sparsity, long seed,
			String probabilityDensityFunction, String pdfParams, int k, 
			CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, boolean replace, String opcode, String istr) {
		super(CPType.Rand, op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.dims = dims;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.minValueStr = minValue;
		this.maxValueStr = maxValue;
		double minDouble, maxDouble;
		try {
			minDouble = !minValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
					Double.valueOf(minValue) : -1;
			maxDouble = !maxValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
					Double.valueOf(maxValue) : -1;
			minMaxAreDoubles = true;
		} catch (NumberFormatException e) {
			// Non double values
			if (!minValueStr.equals(maxValueStr)) {
				throw new DMLRuntimeException("Rand instruction does not support " +
						"non numeric Datatypes for range initializations.");
			}
			minDouble = -1;
			maxDouble = -1;
			minMaxAreDoubles = false;
		}
		this.minValue = minDouble;
		this.maxValue = maxDouble;
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
			CPOperand dims, int rpb, int cpb, String minValue, String maxValue, double sparsity, long seed,
			String probabilityDensityFunction, String pdfParams, int k, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, rpb, cpb, minValue, maxValue, sparsity, seed,
			probabilityDensityFunction, pdfParams, k, null, null, null, false, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			CPOperand dims, int rpb, int cpb, String maxValue, boolean replace, long seed, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, rpb, cpb, "0", maxValue, 1.0, seed,
			null, null, 1, null, null, null, replace, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			CPOperand dims, int rpb, int cpb, CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, rpb, cpb, "0", "1", 1.0, -1,
			null, null, 1, seqFrom, seqTo, seqIncr, false, opcode, istr);
	}
	private DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand out, String opcode, String istr) {
		this(op, mthd, null, out, null, null, null, 0, 0, "0", "0", 0, 0,
			null, null, 1, null, null, null, false, opcode, istr);
	}

	public long getRows() {
		return rows.isLiteral() ? Long.parseLong(rows.getName()) : -1;
	}

	public long getCols() {
		return cols.isLiteral() ? Long.parseLong(cols.getName()) : -1;
	}

	public String getDims() { return dims.getName(); }
	
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
	
	public String getPdf() {
		return pdf;
	}
	
	public String getPdfParams() {
		return pdfParams;
	}
	
	public long getSeed() {
		return seed;
	}
	
	public static DataGenCPInstruction parseInstruction(String str)
	{
		DataGenMethod method = DataGenMethod.INVALID;

		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = s[0];
		
		if ( opcode.equalsIgnoreCase(DataGen.RAND_OPCODE) ) {
			method = DataGenMethod.RAND;
			InstructionUtils.checkNumFields ( s, 13 );
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
		else if ( opcode.equalsIgnoreCase(DataGen.TIME_OPCODE) ) {
			method = DataGenMethod.TIME;
			// 1 operand: outvar
			InstructionUtils.checkNumFields ( s, 1 ); 
		}
		
		CPOperand out = new CPOperand(s[s.length-1]);
		Operator op = null;
		
		if ( method == DataGenMethod.RAND ) 
		{
			CPOperand rows = new CPOperand(s[1]);
			CPOperand cols = new CPOperand(s[2]);
			CPOperand dims = new CPOperand(s[3]);
			int rpb = Integer.parseInt(s[4]);
			int cpb = Integer.parseInt(s[5]);
			double sparsity = !s[8].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
					Double.valueOf(s[8]) : -1;
			long seed = !s[SEED_POSITION_RAND].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
					Long.valueOf(s[SEED_POSITION_RAND]) : -1;
			String pdf = s[10];
			String pdfParams = !s[11].contains( Lop.VARIABLE_NAME_PLACEHOLDER) ?
				s[11] : null;
			int k = Integer.parseInt(s[12]);
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, dims, rpb, cpb, s[6], s[7],
					sparsity, seed, pdf, pdfParams, k, opcode, str);
		}
		else if ( method == DataGenMethod.SEQ) 
		{
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			CPOperand from = new CPOperand(s[5]);
			CPOperand to = new CPOperand(s[6]);
			CPOperand incr = new CPOperand(s[7]);
			
			return new DataGenCPInstruction(op, method, null, out, null, null, null, rpb, cpb, from, to, incr, opcode, str);
		}
		else if ( method == DataGenMethod.SAMPLE) 
		{
			CPOperand rows = new CPOperand(s[2]);
			CPOperand cols = new CPOperand("1", ValueType.INT64, DataType.SCALAR);
			boolean replace = (!s[3].contains(Lop.VARIABLE_NAME_PLACEHOLDER) 
				&& Boolean.valueOf(s[3]));
			
			long seed = Long.parseLong(s[SEED_POSITION_SAMPLE]);
			int rpb = Integer.parseInt(s[5]);
			int cpb = Integer.parseInt(s[6]);
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, null, rpb, cpb, s[1], replace, seed, opcode, str);
		}
		else if ( method == DataGenMethod.TIME) 
		{
			return new DataGenCPInstruction(op, method, out, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
	{
		MatrixBlock soresBlock = null;
		BasicTensor basicTensor = null;
		ScalarObject soresScalar = null;
		
		//process specific datagen operator
		if ( method == DataGenMethod.RAND ) {
			long lrows = ec.getScalarInput(rows).getLongValue();
			long lcols = ec.getScalarInput(cols).getLongValue();
			checkValidDimensions(lrows, lcols);
			
			//generate pseudo-random seed (because not specified) 
			long lSeed = seed; //seed per invocation
			if( lSeed == DataGenOp.UNSPECIFIED_SEED ) {
				if (runtimeSeed == null)
					runtimeSeed = DataGenOp.generateRandomSeed();
				lSeed = runtimeSeed;
			}
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction rand with seed = "+lSeed+".");

			if (output.isTensor()) {
				int[] tDims = DataConverter.getTensorDimensions(ec, dims);
				basicTensor = new BasicTensor(output.getValueType(), tDims);
				basicTensor.allocateDenseBlock();
				if (minValueStr.equals(maxValueStr)) {
					if (minMaxAreDoubles)
						basicTensor.set(minValue);
					else if (output.getValueType() == ValueType.STRING || output.getValueType() == ValueType.BOOLEAN)
							basicTensor.set(minValueStr);
					else {
						throw new DMLRuntimeException("Rand instruction cannot fill numeric "
							+ "tensor with non numeric elements.");
					}
				}
				else {
					// TODO random fill tensor
					lrows = basicTensor.getDim(0);
					lcols = 1;
					for (int d = 1; d < basicTensor.getNumDims(); d++) {
						lcols *= basicTensor.getDim(d);
					}
					RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
							pdf, (int) lrows, (int) lcols, rowsInBlock, colsInBlock, sparsity, minValue, maxValue, pdfParams);
					soresBlock = MatrixBlock.randOperations(rgen, lSeed, numThreads);
					basicTensor.set(soresBlock);
				}
			} else {
				RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
						pdf, (int) lrows, (int) lcols, rowsInBlock, colsInBlock, sparsity, minValue, maxValue, pdfParams);
				soresBlock = MatrixBlock.randOperations(rgen, lSeed, numThreads);
			}
			//reset runtime seed (e.g., when executed in loop)
			runtimeSeed = null;
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
			
			//TODO handle runtime seed
			soresBlock = MatrixBlock.sampleOperations(range, (int)lrows, replace, seed);
		}
		else if ( method == DataGenMethod.TIME ) {
			soresScalar = new IntObject(System.nanoTime());
		}
		
		if( output.isMatrix() ) {
			//guarded sparse block representation change
			if( soresBlock.getInMemorySize() < OptimizerUtils.SAFE_REP_CHANGE_THRES )
				soresBlock.examSparsity();
		
			//release created output
			ec.setMatrixOutput(output.getName(), soresBlock);
		} else if(output.isTensor()) {
			// TODO memory optimization
			ec.setTensorOutput(output.getName(), basicTensor);
		} else if( output.isScalar() )
			ec.setScalarOutput(output.getName(), soresScalar);
	}
	
	private static void checkValidDimensions(long rows, long cols) {
		//check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if( rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE )
			throw new DMLRuntimeException("DataGenCPInstruction does not "
				+ "support dimensions larger than integer: rows="+rows+", cols="+cols+".");
	}

	@Override
	public LineageItem[] getLineageItems(ExecutionContext ec) {
		String tmpInstStr = instString;
		if (getSeed() == DataGenOp.UNSPECIFIED_SEED) {
			//generate pseudo-random seed (because not specified)
			if (runtimeSeed == null)
				runtimeSeed = DataGenOp.generateRandomSeed();
			
			int position = (method == DataGenMethod.RAND) ? SEED_POSITION_RAND + 1 :
					(method == DataGenMethod.SAMPLE) ? SEED_POSITION_SAMPLE + 1 : 0;
			tmpInstStr = InstructionUtils.replaceOperand(
					tmpInstStr, position, String.valueOf(runtimeSeed));
		}
		return new LineageItem[]{new LineageItem(output.getName(), tmpInstStr, getOpcode())};
	}
}
