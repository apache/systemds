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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.FlinkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.flink.functions.ExtractBlockForBinaryReblock;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetAggregateUtils;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class ReblockFLInstruction extends UnaryFLInstruction {
	private int brlen;
	private int bclen;
	private boolean outputEmptyBlocks;

	public ReblockFLInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean emptyBlocks,
								String opcode, String instr) {
		super(op, in, out, opcode, instr);
		brlen = br;
		bclen = bc;
		outputEmptyBlocks = emptyBlocks;
	}

	public static ReblockFLInstruction parseInstruction(String str) throws DMLRuntimeException {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if (opcode.compareTo("rblk") != 0) {
			throw new DMLRuntimeException("Incorrect opcode for ReblockSPInstruction:" + opcode);
		}

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int brlen = Integer.parseInt(parts[3]);
		int bclen = Integer.parseInt(parts[4]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[5]);

		Operator op = null; // no operator for ReblockSPInstruction
		return new ReblockFLInstruction(op, in, out, brlen, bclen, outputEmptyBlocks, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		FlinkExecutionContext flec = (FlinkExecutionContext) ec;

		//set the output characteristics
		MatrixObject mo = flec.getMatrixObject(input1.getName());
		MatrixCharacteristics mc = flec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = flec.getMatrixCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), brlen, bclen, mc.getNonZeros());

		//get the source format form the meta data
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
		if (iimd == null) {
			throw new DMLRuntimeException("Error: Metadata not found");
		}

		//check for in-memory reblock (w/ lazy spark context, potential for latency reduction)
		if (Recompiler.checkCPReblock(flec, input1.getName())) {
			Recompiler.executeInMemoryReblock(flec, input1.getName(), output.getName());
			return;
		}

		if (iimd.getInputInfo() == InputInfo.TextCellInputInfo || iimd.getInputInfo() == InputInfo.MatrixMarketInputInfo) {
			//check jdk version (prevent double.parseDouble contention on <jdk8)
			//flec.checkAndRaiseValidationWarningJDKVersion();

			//get the input textcell dataset
			DataSet<Tuple2<LongWritable, Text>> lines = (DataSet<Tuple2<LongWritable, Text>>)
					flec.getDataSetHandleForVariable(input1.getName(), iimd.getInputInfo());

			//convert textcell to binary block
			DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out =
					DataSetConverterUtils.textCellToBinaryBlock(flec.getFlinkContext(), lines, mcOut,
							outputEmptyBlocks);

			//put output DataSet handle into symbol table
			flec.setDataSetHandleForVariable(output.getName(), out);
			flec.addLineageDataSet(output.getName(), input1.getName());
		} else if (iimd.getInputInfo() == InputInfo.CSVInputInfo) {
			CSVReblockFLInstruction csvInstruction;
			boolean hasHeader = false;
			String delim = ",";
			boolean fill = false;
			double fillValue = 0;
			if (mo.getFileFormatProperties() instanceof CSVFileFormatProperties
					&& mo.getFileFormatProperties() != null) {
				CSVFileFormatProperties props = (CSVFileFormatProperties) mo.getFileFormatProperties();
				hasHeader = props.hasHeader();
				delim = props.getDelim();
				fill = props.isFill();
				fillValue = props.getFillValue();
			}

			csvInstruction = new CSVReblockFLInstruction(null, input1, output, mcOut.getRowsPerBlock(),
					mcOut.getColsPerBlock(),
					hasHeader, delim, fill, fillValue, "csvreblk", instString);
			csvInstruction.processInstruction(flec);
		} else if (iimd.getInputInfo() == InputInfo.BinaryCellInputInfo) {

			DataSet<Tuple2<MatrixIndexes, MatrixCell>> binaryCells = (DataSet<Tuple2<MatrixIndexes, MatrixCell>>) flec.getDataSetHandleForVariable(
					input1.getName(), iimd.getInputInfo());
			DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = DataSetConverterUtils.binaryCellToBinaryBlock(
					flec.getFlinkContext(), binaryCells, mcOut, outputEmptyBlocks);

			//put output DataSet handle into symbol table
			flec.setDataSetHandleForVariable(output.getName(), out);
			flec.addLineageDataSet(output.getName(), input1.getName());
		} else if (iimd.getInputInfo() == InputInfo.BinaryBlockInputInfo) {
			/// HACK ALERT: Workaround for MLContext
			if (mc.getRowsPerBlock() == mcOut.getRowsPerBlock() && mc.getColsPerBlock() == mcOut.getColsPerBlock()) {
				if (mo.getDataSetHandle() != null) {
					DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = (DataSet<Tuple2<MatrixIndexes, MatrixBlock>>) mo.getDataSetHandle().getDataSet();

					//put output DataSet handle into symbol table
					flec.setDataSetHandleForVariable(output.getName(), out);
					flec.addLineageDataSet(output.getName(), input1.getName());
					return;
				} else {
					throw new DMLRuntimeException(
							"Input DataSet is not accessible through buffer pool for ReblockSPInstruction:" + iimd.getInputInfo());
				}
			} else {
				//BINARY BLOCK <- BINARY BLOCK (different sizes)
				DataSet<Tuple2<MatrixIndexes, MatrixBlock>> in1 = flec.getBinaryBlockDataSetHandleForVariable(
						input1.getName());

				DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out =
						in1.flatMap(new ExtractBlockForBinaryReblock(mc, mcOut));
				out = DataSetAggregateUtils.mergeByKey(out);

				//put output DataSet handle into symbol table
				flec.setDataSetHandleForVariable(output.getName(), out);
				flec.addLineageDataSet(output.getName(), input1.getName());
			}
		} else {
			throw new DMLRuntimeException(
					"The given InputInfo is not implemented for ReblockSPInstruction:" + iimd.getInputInfo());
		}

	}
}
