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

package org.apache.sysds.runtime.instructions.cp;

import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DataGenCPInstruction extends UnaryCPInstruction {
	private static final Log LOG = LogFactory.getLog(DataGenCPInstruction.class.getName());
	private OpOpDG method;

	private final CPOperand rows, cols, dims;
	private final int blocksize;
	private boolean minMaxAreDoubles;
	private final String minValueStr, maxValueStr;
	private final double minValue, maxValue, sparsity;
	private final String pdf, pdfParams, frame_data, schema;
	private final long seed;
	private Long runtimeSeed;

	// sequence specific attributes
	private final CPOperand seq_from, seq_to, seq_incr;

	// sample specific attributes
	private final boolean replace;
	private final int numThreads;

	// seed positions
	private static final int SEED_POSITION_RAND = 8;
	private static final int SEED_POSITION_SAMPLE = 4;

	private DataGenCPInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String probabilityDensityFunction, String pdfParams, int k, CPOperand seqFrom, CPOperand seqTo,
		CPOperand seqIncr, boolean replace, String data, String schema, String opcode, String istr) {
		super(CPType.Rand, op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.dims = dims;
		this.blocksize = blen;
		this.minValueStr = minValue;
		this.maxValueStr = maxValue;
		double minDouble, maxDouble;
		try {
			minDouble = !minValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Double.valueOf(minValue) : -1;
			maxDouble = !maxValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Double.valueOf(maxValue) : -1;
			minMaxAreDoubles = true;
		}
		catch(NumberFormatException e) {
			// Non double values
			if(!minValueStr.equals(maxValueStr)) {
				throw new DMLRuntimeException(
					"Rand instruction does not support " + "non numeric Datatypes for range initializations.");
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
		this.frame_data = data;
		this.schema = schema;
	}

	private DataGenCPInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String probabilityDensityFunction, String pdfParams, int k, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, minValue, maxValue, sparsity, seed, probabilityDensityFunction,
			pdfParams, k, null, null, null, false, null, null, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String maxValue, boolean replace, long seed, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "0", maxValue, 1.0, seed, null, null, 1, null, null, null,
			replace, null, null, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "0", "1", 1.0, -1, null, null, 1, seqFrom, seqTo, seqIncr,
			false, null, null, opcode, istr);
	}

	private DataGenCPInstruction(Operator op, OpOpDG mthd, CPOperand out, String opcode, String istr) {
		this(op, mthd, null, out, null, null, null, 0, "0", "0", 0, 0, null, null, 1, null, null, null, false, null,
			null, opcode, istr);
	}

	public DataGenCPInstruction(Operator op, OpOpDG method, CPOperand out, CPOperand rows, CPOperand cols, String data,
		String schema, String opcode, String str) {
		this(op, method, null, out, rows, cols, null, 0, "0", "0", 0, 0, null, null, 1, null, null, null, false, data,
			schema, opcode, str);
	}

	public long getRows() {
		return rows.isLiteral() ? UtilFunctions.parseToLong(rows.getName()) : -1;
	}

	public long getCols() {
		return cols.isLiteral() ? UtilFunctions.parseToLong(cols.getName()) : -1;
	}

	public String getDims() {
		return dims.getName();
	}
	public CPOperand getDimsOperand() { return dims; }

	public int getBlocksize() {
		return blocksize;
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

	public boolean isOnesCol() {
		return minValue == maxValue && minValue == 1 && sparsity == 1 && getCols() == 1;
	}

	public boolean isMatrixCall() {
		return minValue == maxValue && sparsity == 1;
	}

	public long getFrom() {
		return seq_from.isLiteral() ? UtilFunctions.parseToLong(seq_from.getName()) : -1;
	}

	public long getTo() {
		return seq_to.isLiteral() ? UtilFunctions.parseToLong(seq_to.getName()) : -1;
	}

	public long getIncr() {
		return seq_incr.isLiteral() ? UtilFunctions.parseToLong(seq_incr.getName()) : -1;
	}

	public static DataGenCPInstruction parseInstruction(String str) {
		OpOpDG method = null;
		String[] s = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = s[0];

		if(opcode.equalsIgnoreCase(Opcodes.RANDOM.toString())) {
			method = OpOpDG.RAND;
			InstructionUtils.checkNumFields(s, 10, 11);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.SEQUENCE.toString())) {
			method = OpOpDG.SEQ;
			// 8 operands: rows, cols, blen, from, to, incr, outvar
			InstructionUtils.checkNumFields(s, 7);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.SAMPLE.toString())) {
			method = OpOpDG.SAMPLE;
			// 7 operands: range, size, replace, seed, blen, outvar
			InstructionUtils.checkNumFields(s, 6);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.TIME.toString())) {
			method = OpOpDG.TIME;
			// 1 operand: outvar
			InstructionUtils.checkNumFields(s, 1);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.FRAME.toString())) {
			method = OpOpDG.FRAMEINIT;
			InstructionUtils.checkNumFields(s, 5);
		}

		CPOperand out = new CPOperand(s[s.length - 1]);
		Operator op = null;

		if(method == OpOpDG.RAND) {
			int missing; // number of missing params (row & cols or dims)
			CPOperand rows = null, cols = null, dims = null;
			if(s.length == 12) {
				missing = 1;
				rows = new CPOperand(s[1]);
				cols = new CPOperand(s[2]);
			}
			else {
				missing = 2;
				dims = new CPOperand(s[1]);
			}
			int blen = Integer.parseInt(s[4 - missing]);
			double sparsity = !s[7 - missing].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Double
				.parseDouble(s[7 - missing]) : -1;
			long seed = !s[SEED_POSITION_RAND - missing].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Long
				.parseLong(s[SEED_POSITION_RAND - missing]) : -1;
			String pdf = s[9 - missing];
			String pdfParams = !s[10 - missing].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? s[10 - missing] : null;
			int k = Integer.parseInt(s[11 - missing]);

			return new DataGenCPInstruction(op, method, null, out, rows, cols, dims, blen, s[5 - missing],
				s[6 - missing], sparsity, seed, pdf, pdfParams, k, opcode, str);
		}
		else if(method == OpOpDG.SEQ) {
			int blen = Integer.parseInt(s[3]);
			CPOperand from = new CPOperand(s[4]);
			CPOperand to = new CPOperand(s[5]);
			CPOperand incr = new CPOperand(s[6]);

			return new DataGenCPInstruction(op, method, null, out, null, null, null, blen, from, to, incr, opcode, str);
		}
		else if(method == OpOpDG.FRAMEINIT) {
			String data = s[1];
			CPOperand rows = new CPOperand(s[2]);
			CPOperand cols = new CPOperand(s[3]);
			String valueType = s[4];
			return new DataGenCPInstruction(op, method, out, rows, cols, data, valueType, opcode, str);
		}
		else if(method == OpOpDG.SAMPLE) {
			CPOperand rows = new CPOperand(s[2]);
			CPOperand cols = new CPOperand("1", ValueType.INT64, DataType.SCALAR);
			boolean replace = (!s[3].contains(Lop.VARIABLE_NAME_PLACEHOLDER) && Boolean.valueOf(s[3]));
			long seed = !s[SEED_POSITION_SAMPLE].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Long.parseLong(s[SEED_POSITION_SAMPLE]) : -1;
			int blen = Integer.parseInt(s[5]);

			return new DataGenCPInstruction(op, method, null,
				out, rows, cols, null, blen, s[1], replace, seed, opcode, str);
		}
		else if(method == OpOpDG.TIME) {
			return new DataGenCPInstruction(op, method, out, opcode, str);
		}
		else
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		CacheBlock<?> out = null;
		ScalarObject soresScalar = null;

		// process specific datagen operator
		if(method == OpOpDG.RAND)
			out = processRandInstruction(ec);
		else if(method == OpOpDG.SEQ) {
			double lfrom = ec.getScalarInput(seq_from).getDoubleValue();
			double lto = ec.getScalarInput(seq_to).getDoubleValue();
			double lincr = ec.getScalarInput(seq_incr).getDoubleValue();

			// handle default 1 to -1 for special case of from>to
			lincr = LibMatrixDatagen.updateSeqIncr(lfrom, lto, lincr);

			if(LOG.isTraceEnabled())
				LOG.trace(
					"Process DataGenCPInstruction seq with seqFrom=" + lfrom + ", seqTo=" + lto + ", seqIncr" + lincr);

			out = MatrixBlock.seqOperations(lfrom, lto, lincr);
		}
		else if(method == OpOpDG.SAMPLE) {
			long lrows = ec.getScalarInput(rows).getLongValue();
			long range = UtilFunctions.toLong(maxValue);
			checkValidDimensions(lrows, 1);

			if(LOG.isTraceEnabled())
				LOG.trace("Process DataGenCPInstruction sample with range=" + range + ", size=" + lrows + ", replace"
					+ replace + ", seed=" + seed);

			if(range < lrows && !replace)
				throw new DMLRuntimeException("Sample (size=" + lrows + ") larger than population (size=" + range
					+ ") can only be generated with replacement.");

			// TODO handle runtime seed
			out = MatrixBlock.sampleOperations(range, (int) lrows, replace, seed);
		}
		else if(method == OpOpDG.TIME) {
			soresScalar = new IntObject(System.nanoTime());
		}
		else if(method == OpOpDG.FRAMEINIT) {
			int lrows = (int) ec.getScalarInput(rows).getLongValue();
			int lcols = (int) ec.getScalarInput(cols).getLongValue();
			String sparts[] = schema.split(DataExpression.DELIM_NA_STRING_SEP);
			ValueType[] vt = sparts[0].equals(DataExpression.DEFAULT_SCHEMAPARAM) ?
				UtilFunctions.nCopies(lcols, ValueType.STRING) : (sparts.length == 1 && lcols > 1) ?
				UtilFunctions.nCopies(lcols, ValueType.valueOf(sparts[0])) : UtilFunctions.stringToValueType(sparts);
			int schemaLength = vt.length;
			if(schemaLength != lcols)
				throw new DMLRuntimeException("schema-dimension mismatch");

			if(frame_data.equals("")) {
				// TODO fix hard-coded seed, consistently with sparse frame init
				out = UtilFunctions.generateRandomFrameBlock(lrows, lcols, vt, new Random(10));
			}
			else {
				String[] data = frame_data.split(DataExpression.DELIM_NA_STRING_SEP);

				int rowLength = (lrows > 0)?data.length/lrows:0;
				if(data.length != schemaLength && data.length > 1 && rowLength != schemaLength)
					throw new DMLRuntimeException(
						"data values should be equal to number of columns," + " or a single values for all columns");
				out = new FrameBlock(vt);
				FrameBlock outF = (FrameBlock) out;
				if(data.length > 1 && rowLength != schemaLength)  {
					for(int i = 0; i < lrows; i++)
						outF.appendRow(data);
				}
				else if(data.length > 1 && rowLength == schemaLength)
				{
					int beg = 0;
					for(int i = 1; i <= lrows; i++) {
						int end = lcols * i;
						String[] data1 = ArrayUtils.subarray(data, beg, end);
						beg = end;
						outF.appendRow(data1);
					}
				}
				else 
					out = new FrameBlock(vt, frame_data, lrows);
				
			}
		}

		if(output.isScalar())
			ec.setScalarOutput(output.getName(), soresScalar);
		else
			setCacheBlockOutput(ec, out);
	}

	private CacheBlock<?> processRandInstruction(ExecutionContext ec) {
		CacheBlock<?> out;
		long lSeed = generateSeed();

		if(output.isTensor())
			out = processRandInstructionTensor(ec, lSeed);
		else {
			out = processRandInstructionMatrix(ec, lSeed);
		}

		// reset runtime seed (e.g., when executed in loop)
		runtimeSeed = null;
		return out;
	}

	private CacheBlock<?> processRandInstructionMatrix(ExecutionContext ec, long lSeed){

		long lrows = ec.getScalarInput(rows).getLongValue();
		long lcols = ec.getScalarInput(cols).getLongValue();
		checkValidDimensions(lrows, lcols);
		
		if(sparsity == 0.0 && lrows < Integer.MAX_VALUE && lcols < Integer.MAX_VALUE)
			return new MatrixBlock((int)lrows,(int)lcols, 0.0);

		if(ConfigurationManager.isCompressionEnabled() && minValue == maxValue && sparsity == 1.0) {
			// contains constant

			if(lrows > 1000 && lcols > 0 && lrows / lcols > 1)
				return CompressedMatrixBlockFactory.createConstant((int)lrows, (int)lcols, minValue);
			
			return MatrixBlock.randOperations(getGenerator(lrows, lcols), lSeed, numThreads);
		}
		else
			return MatrixBlock.randOperations(getGenerator(lrows, lcols), lSeed, numThreads);
	}

	private CacheBlock<?> processRandInstructionTensor(ExecutionContext ec, long lSeed) {
		int[] tDims = DataConverter.getTensorDimensions(ec, dims);
		CacheBlock<?> out = new TensorBlock(output.getValueType(), tDims).allocateBlock();
		TensorBlock outT = (TensorBlock) out;
		if(minValueStr.equals(maxValueStr)) {
			if(minMaxAreDoubles)
				outT.set(minValue);
			else if(output.getValueType() == ValueType.STRING || output.getValueType() == ValueType.BOOLEAN)
				outT.set(minValueStr);
			else
				throw new DMLRuntimeException("Rand instruction cannot fill numeric tensor with non numeric elements.");
		}
		else {
			long lrows = outT.getDim(0);
			long lcols = 1;
			for(int d = 1; d < outT.getNumDims(); d++) {
				lcols *= outT.getDim(d);
			}
			RandomMatrixGenerator rgen = getGenerator(lrows, lcols);
			MatrixBlock mb = MatrixBlock.randOperations(rgen, lSeed, numThreads);
			outT.set(mb);
		}
		return out;
	}

	private long generateSeed() {
		// generate pseudo-random seed (because not specified)
		long lSeed = seed; // seed per invocation
		if(lSeed == DataGenOp.UNSPECIFIED_SEED) {
			if(runtimeSeed == null)
				runtimeSeed = DataGenOp.generateRandomSeed();
			lSeed = runtimeSeed;
		}

		if(LOG.isTraceEnabled())
			LOG.trace("Process DataGenCPInstruction rand with seed = " + lSeed + ".");

		return lSeed;
	}

	private RandomMatrixGenerator getGenerator(long lrows, long lcols) {
		return LibMatrixDatagen.createRandomMatrixGenerator(pdf,
			(int) lrows,
			(int) lcols,
			blocksize,
			sparsity,
			minValue,
			maxValue,
			pdfParams);
	}

	private void setCacheBlockOutput(ExecutionContext ec, CacheBlock<?> out) {
		if(output.isMatrix()) {
			MatrixBlock outB = (MatrixBlock) out;
			if(out.getInMemorySize() < OptimizerUtils.SAFE_REP_CHANGE_THRES)
				outB.examSparsity();

			// release created output
			LineageItem lin = CacheableData.isBelowCachingThreshold(out) ? null : getLineageItem(ec).getValue();
			ec.setMatrixOutputAndLineage(output.getName(), outB, lin);
		}
		else if(output.isTensor())
			ec.setTensorOutput(output.getName(), (TensorBlock) out);
		else if(output.isFrame())
			ec.setFrameOutput(output.getName(), (FrameBlock) out);
	}

	private static void checkValidDimensions(long rows, long cols) {
		// check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if(rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE)
			throw new DMLRuntimeException("DataGenCPInstruction does not "
				+ "support dimensions larger than integer: rows=" + rows + ", cols=" + cols + ".");
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		String tmpInstStr = instString;

		switch(method) {
			case RAND:
			case SAMPLE: {
				if(getSeed() == DataGenOp.UNSPECIFIED_SEED) {
					// generate pseudo-random seed (because not specified)
					if(runtimeSeed == null)
						runtimeSeed = (minValue == maxValue && sparsity == 1) ? DataGenOp.UNSPECIFIED_SEED : DataGenOp
							.generateRandomSeed();
					int position = (method == OpOpDG.RAND) ? SEED_POSITION_RAND : (method == OpOpDG.SAMPLE) ? SEED_POSITION_SAMPLE : 0;
					tmpInstStr = position != 0 ? InstructionUtils
						.replaceOperand(tmpInstStr, position, String.valueOf(runtimeSeed)) : tmpInstStr;
				}
				// replace output variable name with a placeholder
				tmpInstStr = InstructionUtils.replaceOperandName(tmpInstStr);
				tmpInstStr = method.name().equalsIgnoreCase(
					"rand") ? replaceNonLiteral(tmpInstStr, rows, 2, ec) : replaceNonLiteral(tmpInstStr, rows, 3, ec);
				tmpInstStr = method.name()
					.equalsIgnoreCase("rand") ? replaceNonLiteral(tmpInstStr, cols, 3, ec) : tmpInstStr;
				break;
			}
			case SEQ: {
				// replace output variable name with a placeholder
				tmpInstStr = InstructionUtils.replaceOperandName(tmpInstStr);
				tmpInstStr = replaceNonLiteral(tmpInstStr, seq_from, 5, ec);
				tmpInstStr = replaceNonLiteral(tmpInstStr, seq_to, 6, ec);
				tmpInstStr = replaceNonLiteral(tmpInstStr, seq_incr, 7, ec);
				break;
			}
			case FRAMEINIT: {
				tmpInstStr = InstructionUtils.replaceOperandName(tmpInstStr);
				CPOperand frameInp = new CPOperand(frame_data, ValueType.STRING, DataType.SCALAR, true);
				tmpInstStr = InstructionUtils.replaceOperand(tmpInstStr, 2, frameInp.getLineageLiteral());
				tmpInstStr = replaceNonLiteral(tmpInstStr, rows, 3, ec);
				tmpInstStr = replaceNonLiteral(tmpInstStr, cols, 4, ec);
				CPOperand schemaInp = new CPOperand(schema, ValueType.STRING, DataType.SCALAR, true);
				tmpInstStr = !schema.equalsIgnoreCase("NULL")
						? InstructionUtils.replaceOperand(tmpInstStr, 5, schemaInp.getLineageLiteral()) : tmpInstStr;
			}
			case TIME:
				// only opcode (time) is sufficient to compute from lineage.
				break;
			default:
				throw new DMLRuntimeException("Unsupported datagen op: " + method);
		}
		return Pair.of(output.getName(), new LineageItem(tmpInstStr, getOpcode()));
	}

	private static String replaceNonLiteral(String inst, CPOperand op, int pos, ExecutionContext ec) {
		if(!op.isLiteral())
			inst = InstructionUtils.replaceOperand(inst, pos, new CPOperand(ec.getScalarInput(op)).getLineageLiteral());
		return inst;
	}

}
