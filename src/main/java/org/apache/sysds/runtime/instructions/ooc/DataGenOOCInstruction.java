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
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DataGenOOCInstruction extends UnaryOOCInstruction {

	private final int blen;
	private Types.OpOpDG method;

	// sequence specific attributes
	private final CPOperand seq_from, seq_to, seq_incr;

	public DataGenOOCInstruction(UnaryOperator op, Types.OpOpDG mthd, CPOperand in, CPOperand out, int blen, CPOperand seqFrom,
		CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		super(OOCType.Rand, op, in, out, opcode, istr);
		this.blen = blen;
		this.method = mthd;
		this.seq_from = seqFrom;
		this.seq_to = seqTo;
		this.seq_incr = seqIncr;
	}

	public static DataGenOOCInstruction parseInstruction(String str) {
		Types.OpOpDG method = null;
		String[] s = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = s[0];

		if(opcode.equalsIgnoreCase(Opcodes.SEQUENCE.toString())) {
			method = Types.OpOpDG.SEQ;
			// 8 operands: rows, cols, blen, from, to, incr, outvar
			InstructionUtils.checkNumFields(s, 7);
		}
		else
			throw new NotImplementedException(); // TODO

		CPOperand out = new CPOperand(s[s.length - 1]);
		UnaryOperator op = null;

		if(method == Types.OpOpDG.SEQ) {
			int blen = Integer.parseInt(s[3]);
			CPOperand from = new CPOperand(s[4]);
			CPOperand to = new CPOperand(s[5]);
			CPOperand incr = new CPOperand(s[6]);

			return new DataGenOOCInstruction(op, method, null, out, blen, from, to, incr, opcode, str);
		}
		else
			throw new NotImplementedException();
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		final OOCStream<IndexedMatrixValue> qOut = createWritableStream();

		// process specific datagen operator
		if(method == Types.OpOpDG.SEQ) {
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
}
