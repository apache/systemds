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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.data.IndexedTensorBlock;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import scala.Tuple2;

import java.util.Iterator;
import java.util.List;

public class MatrixReshapeSPInstruction extends UnarySPInstruction
{
	private final CPOperand _opRows;
	private final CPOperand _opCols;
	private final CPOperand _opByRow;
	private final boolean _outputEmptyBlocks;

	private MatrixReshapeSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
			CPOperand out, boolean outputEmptyBlocks, String opcode, String istr) {
		super(SPType.MatrixReshape, op, in1, out, opcode, istr);
		_opRows = in2;
		_opCols = in3;
		_opByRow = in4;
		_outputEmptyBlocks = outputEmptyBlocks;
	}

	public static MatrixReshapeSPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields( parts, 7 );
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand rows = new CPOperand(parts[2]);
		CPOperand cols = new CPOperand(parts[3]);
		//TODO handle dims for tensors parts[4]
		CPOperand byRow = new CPOperand(parts[5]);
		CPOperand out = new CPOperand(parts[6]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[7]);
		 
		if(!opcode.equalsIgnoreCase(Opcodes.RESHAPE.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeInstruction: " + str);
		else
			return new MatrixReshapeSPInstruction(new Operator(true), in1, rows, cols, byRow, out, outputEmptyBlocks, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get parameters
		long rows = ec.getScalarInput(_opRows).getLongValue(); //save cast
		long cols = ec.getScalarInput(_opCols).getLongValue(); //save cast
		boolean byRow = ec.getScalarInput(_opByRow.getName(), ValueType.BOOLEAN, _opByRow.isLiteral()).getBooleanValue();

		DataCharacteristics mcIn = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if (input1.getDataType() == Types.DataType.MATRIX) {
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec
					.getBinaryMatrixBlockRDDHandleForVariable(input1.getName(), -1, _outputEmptyBlocks);

			//update output characteristics and sanity check
			mcOut.set(rows, cols, mcIn.getBlocksize(), mcIn.getNonZeros());
			if (!mcIn.nnzKnown())
				mcOut.setNonZerosBound(mcIn.getNonZerosBound());
			if (mcIn.getRows() * mcIn.getCols() != mcOut.getRows() * mcOut.getCols()) {
				throw new DMLRuntimeException("Incompatible matrix characteristics for reshape: "
						+ mcIn.getRows() + "x" + mcIn.getCols() + " vs " + mcOut.getRows() + "x" + mcOut.getCols());
			}

			if (!_outputEmptyBlocks)
				in1 = in1.filter(new FilterNonEmptyBlocksFunction());

			//execute reshape instruction
			JavaPairRDD<MatrixIndexes, MatrixBlock> out =
					in1.flatMapToPair(new RDDReshapeFunction(mcIn, mcOut, byRow, _outputEmptyBlocks));
			out = RDDAggregateUtils.mergeByKey(out);

			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		} else {
			// TODO Tensor reshape
			JavaPairRDD<TensorIndexes, TensorBlock> in1 = sec.getBinaryTensorBlockRDDHandleForVariable(input1.getName(),
					-1, _outputEmptyBlocks);
			JavaPairRDD<TensorIndexes, TensorBlock> out = in1.flatMapToPair(
					new RDDTensorReshapeFunction(mcIn, mcOut, byRow, _outputEmptyBlocks));
			// TODO merge by key
			//out = RDDAggregateUtils.mergeByKey(out);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
	}

	public CPOperand getOpRows() {
		return _opRows;
	}

	public CPOperand getOpCols() {
		return _opCols;
	}

	public CPOperand getOpByRow() {
		return _opByRow;
	}

	private static class RDDReshapeFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 2819309412002224478L;
		
		private final DataCharacteristics _mcIn;
		private final DataCharacteristics _mcOut;
		private final boolean _byrow;
		private final boolean _outputEmptyBlocks;
		
		public RDDReshapeFunction(DataCharacteristics mcIn, DataCharacteristics mcOut, boolean byrow, boolean outputEmptyBlocks) {
			_mcIn = mcIn;
			_mcOut = mcOut;
			_byrow = byrow;
			_outputEmptyBlocks = outputEmptyBlocks;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			//input conversion (for libmatrixreorg compatibility)
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(arg0);
			
			//execute actual reshape operation
			List<IndexedMatrixValue> out = LibMatrixReorg
				.reshape(in, _mcIn, _mcOut, _byrow, _outputEmptyBlocks);

			//output conversion (for compatibility w/ rdd schema)
			return SparkUtils.fromIndexedMatrixBlock(out).iterator();
		}
	}

	@SuppressWarnings("unused")
	private static class RDDTensorReshapeFunction implements PairFlatMapFunction<Tuple2<TensorIndexes, TensorBlock>,
			TensorIndexes, TensorBlock> {
		private static final long serialVersionUID = 8030648988828223639L;

		private final DataCharacteristics _mcIn;
		private final DataCharacteristics _mcOut;
		private final boolean _byrow;
		private final boolean _outputEmptyBlocks;

		public RDDTensorReshapeFunction(DataCharacteristics mcIn, DataCharacteristics mcOut, boolean byrow, boolean outputEmptyBlocks) {
			_mcIn = mcIn;
			_mcOut = mcOut;
			_byrow = byrow;
			_outputEmptyBlocks = outputEmptyBlocks;
		}

		@Override
		public Iterator<Tuple2<TensorIndexes, TensorBlock>> call(Tuple2<TensorIndexes, TensorBlock> arg0)
				throws Exception {
			//input conversion (for libmatrixreorg compatibility)
			IndexedTensorBlock in = SparkUtils.toIndexedTensorBlock(arg0);

			//execute actual reshape operation
			//LibTensorReorg.reshape()
//			List<IndexedTensorBlock> out = LibTensorReorg
//					.reshape(in, _mcIn, _mcOut, _byrow, _outputEmptyBlocks);
//			// TODO create iterator
			return null;
		}
	}
}
