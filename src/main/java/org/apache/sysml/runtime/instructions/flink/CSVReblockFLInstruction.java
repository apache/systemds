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

package org.apache.sysml.runtime.instructions.flink;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.FlinkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class CSVReblockFLInstruction extends UnaryFLInstruction {

	private int _brlen;
	private int _bclen;
	private boolean _hasHeader;
	private String _delim;
	private boolean _fill;
	private double _missingValue;

	public CSVReblockFLInstruction(Operator op, CPOperand in, CPOperand out,
								   int br, int bc, boolean hasHeader, String delim, boolean fill,
								   double missingValue, String opcode, String instr) {
		super(op, in, out, opcode, instr);
		_brlen = br;
		_bclen = bc;
		_hasHeader = hasHeader;
		_delim = delim;
		_fill = fill;
		_missingValue = missingValue;
	}

	public static CSVReblockFLInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		String opcode = InstructionUtils.getOpCode(str);
		if (!opcode.equals("csvrblk")) {
			throw new DMLRuntimeException(
					"Incorrect opcode for CSVReblockSPInstruction:" + opcode);
		}

		// Example parts of CSVReblockSPInstruction:
		// [csvrblk, pREADmissing_val_maps·MATRIX·DOUBLE, _mVar37·MATRIX·DOUBLE,
		// 1000, 1000, false, ,, true, 0.0]
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int brlen = Integer.parseInt(parts[3]);
		int bclen = Integer.parseInt(parts[4]);
		boolean hasHeader = Boolean.parseBoolean(parts[5]);
		String delim = parts[6];
		boolean fill = Boolean.parseBoolean(parts[7]);
		double missingValue = Double.parseDouble(parts[8]);

		return new CSVReblockFLInstruction(null, in, out, brlen, bclen,
				hasHeader, delim, fill, missingValue, opcode, str);
	}

	@Override
	@SuppressWarnings("unchecked")
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		FlinkExecutionContext flec = (FlinkExecutionContext) ec;

		//sanity check input info
		MatrixObject mo = flec.getMatrixObject(input1.getName());
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
		if (iimd.getInputInfo() != InputInfo.CSVInputInfo) {
			throw new DMLRuntimeException("The given InputInfo is not implemented for "
					+ "CSVReblockSPInstruction:" + iimd.getInputInfo());
		}

		// set output characteristics
		MatrixCharacteristics mcIn = flec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = flec.getMatrixCharacteristics(output.getName());
		mcOut.set(mcIn.getRows(), mcIn.getCols(), _brlen, _bclen);

		//check for in-memory reblock
		if (Recompiler.checkCPReblock(flec, input1.getName())) {
			Recompiler.executeInMemoryReblock(flec, input1.getName(), output.getName());
			return;
		}

		// get dataset handle
		DataSet<Tuple2<Integer, String>> in = (DataSet<Tuple2<Integer, String>>) flec.getDataSetHandleForVariable(
				input1.getName(), iimd.getInputInfo());

		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = DataSetConverterUtils.csvToBinaryBlock(
				flec.getFlinkContext(), in, mcOut, _hasHeader, _delim, _fill, _missingValue);

		// put output DataSet handle into symbol table
		flec.setDataSetHandleForVariable(output.getName(), out);
		flec.addLineageDataSet(output.getName(), input1.getName());
	}
}
