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
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import scala.Tuple2;

public class AppendGAlignedSPInstruction extends AppendSPInstruction {
	private AppendGAlignedSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		boolean cbind, String opcode, String istr) {
		super(SPType.GAppend, op, in1, in2, out, cbind, opcode, istr);
	}

	public static AppendGAlignedSPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 5);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		boolean cbind = Boolean.parseBoolean(parts[5]);
		
		if(!opcode.equalsIgnoreCase(Opcodes.GALIGNEDAPPEND.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendGSPInstruction: " + str);
		
		return new AppendGAlignedSPInstruction(
				new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
				in1, in2, in3, out, cbind, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		// general case append (map-extend, aggregate)
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		checkBinaryAppendInputCharacteristics(sec, _cbind, false, true);
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		
		// Simple changing of matrix indexes of RHS
		long shiftBy = _cbind ? mc1.getNumColBlocks() : mc1.getNumRowBlocks();
		out = in2.mapToPair(new ShiftColumnIndex(shiftBy, _cbind));
		out = in1.union( out );
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputDataCharacteristics(sec, _cbind);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}

	public static class ShiftColumnIndex implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -5185023611319654242L;
		
		private long _shiftBy;
		private boolean _cbind;
		
		public ShiftColumnIndex(long shiftBy, boolean cbind) {
			_shiftBy = shiftBy;
			_cbind = cbind;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{	
			long rix = _cbind ? kv._1.getRowIndex() : kv._1.getRowIndex() + _shiftBy;
			long cix = _cbind ? kv._1.getColumnIndex() + _shiftBy : kv._1.getColumnIndex();
			return new Tuple2<>(new MatrixIndexes(rix, cix), kv._2);
		}
	}
}