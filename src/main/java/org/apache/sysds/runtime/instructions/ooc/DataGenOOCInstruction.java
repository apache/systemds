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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DataGenOOCInstruction extends UnaryOOCInstruction {
	private static final Log LOG = LogFactory.getLog(DataGenOOCInstruction.class.getName());
	private Types.OpOpDG method;

	private final CPOperand rows, cols, dims;
	private final int blen;
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

	private DataGenOOCInstruction(UnaryOperator op, Types.OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String probabilityDensityFunction, String pdfParams, int k, CPOperand seqFrom, CPOperand seqTo,
		CPOperand seqIncr, boolean replace, String data, String schema, String opcode, String istr) {
		super(OOCInstruction.OOCType.Rand, op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.dims = dims;
		this.blen = blen;
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

	private DataGenOOCInstruction(UnaryOperator op, Types.OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "0", "1", 1.0, -1, null, null, 1, seqFrom, seqTo, seqIncr,
			false, null, null, opcode, istr);
	}

	private DataGenOOCInstruction(UnaryOperator op, Types.OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String probabilityDensityFunction, String pdfParams, int k, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, minValue, maxValue, sparsity, seed, probabilityDensityFunction,
			pdfParams, k, null, null, null, false, null, null, opcode, istr);
	}

	public static DataGenOOCInstruction parseInstruction(String str) {
		Types.OpOpDG method = null;
		String[] s = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = s[0];

		if(opcode.equalsIgnoreCase(Opcodes.RANDOM.toString())) {
			method = Types.OpOpDG.RAND;
			InstructionUtils.checkNumFields(s, 10, 11);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.SEQUENCE.toString())) {
			method = Types.OpOpDG.SEQ;
			// 8 operands: rows, cols, blen, from, to, incr, outvar
			InstructionUtils.checkNumFields(s, 7);
		}
		else
			throw new NotImplementedException(); // TODO

		CPOperand out = new CPOperand(s[s.length - 1]);
		UnaryOperator op = null;

		if(method == Types.OpOpDG.RAND) {
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

			return new DataGenOOCInstruction(op, method, null, out, rows, cols, dims, blen, s[5 - missing],
				s[6 - missing], sparsity, seed, pdf, pdfParams, k, opcode, str);
		}
		else if(method == Types.OpOpDG.SEQ) {
			int blen = Integer.parseInt(s[3]);
			CPOperand from = new CPOperand(s[4]);
			CPOperand to = new CPOperand(s[5]);
			CPOperand incr = new CPOperand(s[6]);

			return new DataGenOOCInstruction(op, method, null, out, null, null, null, blen, from, to, incr, opcode, str);
		}
		else
			throw new NotImplementedException();
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		final OOCStream<IndexedMatrixValue> qOut = createWritableStream();

		// process specific datagen operator
		if (method == Types.OpOpDG.RAND) {
			if (!output.isMatrix())
				throw new NotImplementedException();

			long lSeed = generateSeed();
			long lrows = ec.getScalarInput(rows).getLongValue();
			long lcols = ec.getScalarInput(cols).getLongValue();
			checkValidDimensions(lrows, lcols);

			if (!pdf.equalsIgnoreCase("uniform") || minValue != maxValue)
				throw new NotImplementedException(); // TODO modified version of rng as in LibMatrixDatagen to handle blocks independently

			OOCStream<MatrixIndexes> qIn = createWritableStream();
			int nrb = (int)((lrows-1) / blen)+1;
			int ncb = (int)((lcols-1) / blen)+1;

			for (int row = 0; row < nrb; row++)
				for (int col = 0; col < ncb; col++)
					qIn.enqueue(new MatrixIndexes(row+1, col+1));

			qIn.closeInput();

			if(sparsity == 0.0 && lrows < Integer.MAX_VALUE && lcols < Integer.MAX_VALUE) {
				mapOOC(qIn, qOut, idx -> {
					long rlen = Math.min(blen, lrows - (idx.getRowIndex()-1) * blen);
					long clen =  Math.min(blen, lcols - (idx.getColumnIndex()-1) * blen);
					return new IndexedMatrixValue(idx, new MatrixBlock((int)rlen, (int)clen, 0.0));
				});
				return;
			}

			mapOOC(qIn, qOut, idx -> {
				long rlen = Math.min(blen, lrows - (idx.getRowIndex()-1) * blen);
				long clen =  Math.min(blen, lcols - (idx.getColumnIndex()-1) * blen);
				MatrixBlock mout = MatrixBlock.randOperations(getGenerator(rlen, clen), lSeed);
				return new IndexedMatrixValue(idx, mout);
			});
		}
		else if(method == Types.OpOpDG.SEQ) {
			double lfrom = ec.getScalarInput(seq_from).getDoubleValue();
			double lto = ec.getScalarInput(seq_to).getDoubleValue();
			double lincr = ec.getScalarInput(seq_incr).getDoubleValue();

			// handle default 1 to -1 for special case of from>to
			lincr = LibMatrixDatagen.updateSeqIncr(lfrom, lto, lincr);

			if(LOG.isTraceEnabled())
				LOG.trace(
					"Process DataGenOOCInstruction seq with seqFrom=" + lfrom + ", seqTo=" + lto + ", seqIncr" + lincr);

			final int maxK = (int) UtilFunctions.getSeqLength(lfrom, lto, lincr);
			final double finalLincr = lincr;


			submitOOCTask(() -> {
				int k = 0;
				double curFrom = lfrom;
				double curTo;
				MatrixBlock mb;

				while (k < maxK) {
					long desiredLen = Math.min(blen, maxK - k);
					curTo = curFrom + (desiredLen - 1) * finalLincr;
					long actualLen = UtilFunctions.getSeqLength(curFrom, curTo, finalLincr);

					if (actualLen != desiredLen) {
						// Then we add / subtract a small correction term
						curTo += (actualLen < desiredLen) ? finalLincr / 2 : -finalLincr / 2;

						if (UtilFunctions.getSeqLength(curFrom, curTo, finalLincr) != desiredLen)
							throw new DMLRuntimeException("OOC seq could not construct the right number of elements.");
					}

					mb = MatrixBlock.seqOperations(curFrom, curTo, finalLincr);
					qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(1 + k / blen, 1), mb));
					curFrom = mb.get(mb.getNumRows() - 1, 0) + finalLincr;
					k += blen;
				}

				qOut.closeInput();
			}, qOut);
		}
		else
			throw new NotImplementedException();

		ec.getMatrixObject(output).setStreamHandle(qOut);
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
			LOG.trace("Process DataGenOOCInstruction rand with seed = " + lSeed + ".");

		return lSeed;
	}

	private static void checkValidDimensions(long rows, long cols) {
		// check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if(rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE)
			throw new DMLRuntimeException("DataGenOOCInstruction does not "
				+ "support dimensions larger than integer: rows=" + rows + ", cols=" + cols + ".");
	}

	private RandomMatrixGenerator getGenerator(long lrows, long lcols) {
		return LibMatrixDatagen.createRandomMatrixGenerator(pdf,
			(int) lrows,
			(int) lcols,
			blen,
			sparsity,
			minValue,
			maxValue,
			pdfParams);
	}
}
