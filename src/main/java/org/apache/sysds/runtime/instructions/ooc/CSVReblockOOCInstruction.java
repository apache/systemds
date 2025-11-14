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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.ReaderTextCSVParallel;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class CSVReblockOOCInstruction extends ComputationOOCInstruction {
	private final int blen;

	private CSVReblockOOCInstruction(Operator op, CPOperand in, CPOperand out, int blocklength, String opcode,
		String instr) {
		super(OOCType.Reblock, op, in, out, opcode, instr);
		blen = blocklength;
	}

	public static CSVReblockOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equals(Opcodes.CSVRBLK.toString()))
			throw new DMLRuntimeException("Incorrect opcode for CSVReblockOOCInstruction:" + opcode);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen = Integer.parseInt(parts[3]);
		return new CSVReblockOOCInstruction(null, in, out, blen, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject min = ec.getMatrixObject(input1);
		DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), blen, mc.getNonZeros());

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		addOutStream(qOut);

		FileFormatProperties props = min.getFileFormatProperties();
		final FileFormatPropertiesCSV csvProps = props instanceof FileFormatPropertiesCSV ? (FileFormatPropertiesCSV) props
			: new FileFormatPropertiesCSV();

		final ReaderTextCSVParallel reader = new ReaderTextCSVParallel(csvProps);
		final String fileName = min.getFileName();
		final long rows = mc.getRows();
		final long cols = mc.getCols();
		final long nnz = mc.getNonZeros();

		submitOOCTask(() -> {
			try {
				reader.readMatrixAsStream(qOut, fileName, rows, cols, blen, nnz);
			}
			catch(Exception ex) {
				throw (ex instanceof DMLRuntimeException) ? (DMLRuntimeException) ex : new DMLRuntimeException(ex);
			}
		}, qOut);

		MatrixObject mout = ec.getMatrixObject(output);
		mout.setStreamHandle(qOut);
	}
}
