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

package org.tugraz.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.lops.UnaryCP;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public class CastSPInstruction extends UnarySPInstruction {

	private CastSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		super(SPType.Cast, op, in, out, opcode, istr);
	}

	public static CastSPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		return new CastSPInstruction(null, in, out, opcode, str);
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		//get input RDD and prepare output
		JavaPairRDD<?,?> in = sec.getRDDHandleForVariable(input1.getName(), InputInfo.BinaryBlockInputInfo, -1, true);
		DataCharacteristics mcIn = sec.getDataCharacteristics( input1.getName() );
		JavaPairRDD<?,?> out = null;
		
		//convert frame-matrix / matrix-frame and set output
		if( opcode.equals(UnaryCP.CAST_AS_MATRIX_OPCODE) ) {
			DataCharacteristics mcOut = new MatrixCharacteristics(mcIn);
			mcOut.setBlocksize(ConfigurationManager.getBlocksize());
			out = FrameRDDConverterUtils.binaryBlockToMatrixBlock(
				(JavaPairRDD<Long, FrameBlock>)in, mcIn, mcOut);
		}
		else if( opcode.equals(UnaryCP.CAST_AS_FRAME_OPCODE) ) {
			out = FrameRDDConverterUtils.matrixBlockToBinaryBlockLongIndex(sec.getSparkContext(), 
				(JavaPairRDD<MatrixIndexes, MatrixBlock>)in, mcIn);
		}
		else {
			throw new DMLRuntimeException("Unsupported spark cast operation: "+opcode);
		}
		
		//update output statistics and add lineage
		sec.setRDDHandleForVariable(output.getName(), out);
		updateUnaryOutputDataCharacteristics(sec, input1.getName(), output.getName());
		sec.addLineageRDD(output.getName(), input1.getName());
		
		//update schema information for output frame
		if( opcode.equals(UnaryCP.CAST_AS_FRAME_OPCODE) ) {
			sec.getFrameObject(output.getName()).setSchema(
				UtilFunctions.nCopies((int)mcIn.getCols(), ValueType.FP64));
		}
	}
}
