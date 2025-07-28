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

package org.apache.sysds.runtime.instructions;

import java.util.HashMap;

import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.AggregateBinaryGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.AggregateUnaryGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.ArithmeticBinaryGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.BuiltinBinaryGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.BuiltinUnaryGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.DnnGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.MMTSJGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.MatrixAppendGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.MatrixIndexingGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.MatrixMatrixAxpyGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.MatrixReshapeGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.RelationalBinaryGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.ReorgGPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction.GPUINSTRUCTION_TYPE;
import org.apache.sysds.runtime.instructions.gpu.SpoofCUDAInstruction;

public class GPUInstructionParser  extends InstructionParser 
{
	static final HashMap<String, GPUINSTRUCTION_TYPE> String2GPUInstructionType;
	static {
		String2GPUInstructionType = new HashMap<>();

		// Neural Network Operators
		String2GPUInstructionType.put( "relu_backward",          GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "conv2d",                 GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "conv2d_bias_add",        GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "conv2d_backward_filter", GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "conv2d_backward_data",   GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "maxpooling",             GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "maxpooling_backward",    GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "avgpooling",             GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "avgpooling_backward",    GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "bias_add",               GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "bias_multiply",          GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "channel_sums",          GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "lstm",                 	GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "lstm_backward",         GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "batch_norm2d",           GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "batch_norm2d_backward",  GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "batch_norm2d_test",      GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "batch_norm2d_train",      GPUINSTRUCTION_TYPE.Dnn);
		String2GPUInstructionType.put( "update_nesterov_x",      GPUINSTRUCTION_TYPE.Dnn);
		
		// Matrix Multiply Operators
		String2GPUInstructionType.put( "ba+*",  GPUINSTRUCTION_TYPE.AggregateBinary);
		String2GPUInstructionType.put( "tsmm",  GPUINSTRUCTION_TYPE.MMTSJ);

		// Reorg/Transpose
		String2GPUInstructionType.put( "r'",    GPUINSTRUCTION_TYPE.Reorg);
		String2GPUInstructionType.put( "rshape",GPUINSTRUCTION_TYPE.MatrixReshape);

		// Matrix Manipulation
		String2GPUInstructionType.put( "append", GPUINSTRUCTION_TYPE.Append);

		// Binary Cellwise
		String2GPUInstructionType.put( "+",    GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "-",    GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "*",    GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "/",    GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "%%",   GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "%/%",  GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "^",    GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "1-*",  GPUINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2GPUInstructionType.put( "^2",   GPUINSTRUCTION_TYPE.ArithmeticBinary); //special ^ case
		String2GPUInstructionType.put( "*2",   GPUINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2GPUInstructionType.put( "-nz",  GPUINSTRUCTION_TYPE.ArithmeticBinary); //special - case
		String2GPUInstructionType.put( "+*",   GPUINSTRUCTION_TYPE.ArithmeticBinary);
		String2GPUInstructionType.put( "-*",   GPUINSTRUCTION_TYPE.ArithmeticBinary);
		
		// Unary Builtin functions
		String2GPUInstructionType.put( "exp",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "log",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "abs",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "sqrt",  GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "round", GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "floor", GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "ceil",  GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "sin",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "cos",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "tan",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "sinh",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "cosh",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "tanh",   GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "asin",  GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "acos",  GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "atan",  GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "sign",  GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "sigmoid", GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "softmax", GPUINSTRUCTION_TYPE.BuiltinUnary);

		// Binary Builtin functions
		String2GPUInstructionType.put( "solve", GPUINSTRUCTION_TYPE.BuiltinBinary);
		String2GPUInstructionType.put( "min", GPUINSTRUCTION_TYPE.BuiltinBinary);
		String2GPUInstructionType.put( "max", GPUINSTRUCTION_TYPE.BuiltinBinary);

		// Aggregate Unary
		String2GPUInstructionType.put( "ua+"     , GPUINSTRUCTION_TYPE.AggregateUnary); // Sum
		String2GPUInstructionType.put( "uak+"    , GPUINSTRUCTION_TYPE.AggregateUnary); // Sum
		String2GPUInstructionType.put( "uar+"    , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Sum
		String2GPUInstructionType.put( "uark+"   , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Sum
		String2GPUInstructionType.put( "uac+"    , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Sum
		String2GPUInstructionType.put( "uack+"   , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Sum
		String2GPUInstructionType.put( "ua*"     , GPUINSTRUCTION_TYPE.AggregateUnary); // Multiplication
		String2GPUInstructionType.put( "uamean"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Mean
		String2GPUInstructionType.put( "uarmean" , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Mean
		String2GPUInstructionType.put( "uacmean" , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Mean
		String2GPUInstructionType.put( "uamax"   , GPUINSTRUCTION_TYPE.AggregateUnary); // Max
		String2GPUInstructionType.put( "uarmax"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Max
		String2GPUInstructionType.put( "uacmax"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Max
		String2GPUInstructionType.put( "uamin"   , GPUINSTRUCTION_TYPE.AggregateUnary); // Min
		String2GPUInstructionType.put( "uarmin"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Min
		String2GPUInstructionType.put( "uacmin"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Min
		String2GPUInstructionType.put( "uasqk+"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Sum of Squares
		String2GPUInstructionType.put( "uarsqk+" , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Sum of Squares
		String2GPUInstructionType.put( "uacsqk+" , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Sum of Squares
		String2GPUInstructionType.put( "uavar"   , GPUINSTRUCTION_TYPE.AggregateUnary); // Variance
		String2GPUInstructionType.put( "uarvar"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Row Variance
		String2GPUInstructionType.put( "uacvar"  , GPUINSTRUCTION_TYPE.AggregateUnary); // Col Variance

		// Cumulative Ops
		String2GPUInstructionType.put( "ucumk+"  , GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "urowcumk+", GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "ucum*"   , GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "ucumk+*" , GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "ucummin" , GPUINSTRUCTION_TYPE.BuiltinUnary);
		String2GPUInstructionType.put( "ucummax" , GPUINSTRUCTION_TYPE.BuiltinUnary);
		// Relational Binary
		String2GPUInstructionType.put( "=="   , GPUINSTRUCTION_TYPE.RelationalBinary);
		String2GPUInstructionType.put( "!="   , GPUINSTRUCTION_TYPE.RelationalBinary);
		String2GPUInstructionType.put( "<"    , GPUINSTRUCTION_TYPE.RelationalBinary);
		String2GPUInstructionType.put( ">"    , GPUINSTRUCTION_TYPE.RelationalBinary);
		String2GPUInstructionType.put( "<="   , GPUINSTRUCTION_TYPE.RelationalBinary);
		String2GPUInstructionType.put( ">="   , GPUINSTRUCTION_TYPE.RelationalBinary);
		
		// Indexing 
		String2GPUInstructionType.put( RightIndex.OPCODE, GPUINSTRUCTION_TYPE.MatrixIndexing);

		String2GPUInstructionType.put( "spoof"   , GPUINSTRUCTION_TYPE.SpoofFused);
	}
	
	public static GPUInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		GPUINSTRUCTION_TYPE cptype = InstructionUtils.getGPUType(str);
		if ( cptype == null ) 
			throw new DMLRuntimeException("Unable derive cptype for instruction: " + str);
		GPUInstruction cpinst = parseSingleInstruction(cptype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static GPUInstruction parseSingleInstruction ( GPUINSTRUCTION_TYPE gputype, String str ) {
		if( str == null || str.isEmpty() ) 
			return null;	
		if( gputype == null )
			throw new DMLRuntimeException("The instruction is not GPU-enabled:" + str);
		
		switch(gputype) {
			case AggregateUnary:
				return AggregateUnaryGPUInstruction.parseInstruction(str);

			case AggregateBinary:
				return AggregateBinaryGPUInstruction.parseInstruction(str);
			
			case BuiltinUnary:
				return BuiltinUnaryGPUInstruction.parseInstruction(str);

			case BuiltinBinary:
				return BuiltinBinaryGPUInstruction.parseInstruction(str);

			case Append:
				return MatrixAppendGPUInstruction.parseInstruction(str);

			case Dnn:
				return DnnGPUInstruction.parseInstruction(str);
				
			case MMTSJ:
				return MMTSJGPUInstruction.parseInstruction(str);
				
			case Reorg:
				return ReorgGPUInstruction.parseInstruction(str);
				
			case MatrixReshape:
				return MatrixReshapeGPUInstruction.parseInstruction(str);
				
			case ArithmeticBinary:
				String opcode = InstructionUtils.getOpCode(str);
				if( opcode.equals("+*") || opcode.equals("-*")  )
					return MatrixMatrixAxpyGPUInstruction.parseInstruction(str);
				else
					return ArithmeticBinaryGPUInstruction.parseInstruction(str);
			case RelationalBinary:
				return RelationalBinaryGPUInstruction.parseInstruction(str);

			case MatrixIndexing:
				return MatrixIndexingGPUInstruction.parseInstruction(str);

			case SpoofFused:
				return SpoofCUDAInstruction.parseInstruction(str);

			default: 
				throw new DMLRuntimeException("Invalid GPU Instruction Type: " + gputype );
		}
	}	
}
