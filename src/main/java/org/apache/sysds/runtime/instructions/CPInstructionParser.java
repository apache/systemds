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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.InstructionType;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AppendCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BroadcastCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BuiltinNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CompressionCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DeCompressionCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DnnCPInstruction;
import org.apache.sysds.runtime.instructions.cp.EvictCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.LocalCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnComplexMatrixBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.PMMJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.PrefetchCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantilePickCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantileSortCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReshapeCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SpoofCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SqlCPInstruction;
import org.apache.sysds.runtime.instructions.cp.StringInitCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UaggOuterChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.cpfile.MatrixIndexingCPFileInstruction;

public class CPInstructionParser extends InstructionParser {
	protected static final Log LOG = LogFactory.getLog(CPInstructionParser.class.getName());

	public static CPInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		InstructionType cptype = InstructionUtils.getCPType(str);
		if ( cptype == null )
			throw new DMLRuntimeException("Unable derive cptype for instruction: " + str);
		CPInstruction cpinst = parseSingleInstruction(cptype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static CPInstruction parseSingleInstruction ( InstructionType cptype, String str ) {
		ExecType execType;
		if ( str == null || str.isEmpty() ) 
			return null;
		switch(cptype) {
			case AggregateUnary:
				return AggregateUnaryCPInstruction.parseInstruction(str);
			
			case AggregateBinary:
				return AggregateBinaryCPInstruction.parseInstruction(str);
	
			case AggregateTernary:
				return AggregateTernaryCPInstruction.parseInstruction(str);
			
			case Unary:
				return UnaryCPInstruction.parseInstruction(str);

			case Binary:
				return BinaryCPInstruction.parseInstruction(str);
			
			case Ternary:
				return TernaryCPInstruction.parseInstruction(str);
			
			case Quaternary:
				return QuaternaryCPInstruction.parseInstruction(str);
			
			case BuiltinNary:
				return BuiltinNaryCPInstruction.parseInstruction(str);
			
			case Ctable:
				return CtableCPInstruction.parseInstruction(str);
			
			case Reorg:
				return ReorgCPInstruction.parseInstruction(str);
				
			case Dnn:
				 return DnnCPInstruction.parseInstruction(str);
				
			case UaggOuterChain:
				return UaggOuterChainCPInstruction.parseInstruction(str);
				
			case Reshape:
				return ReshapeCPInstruction.parseInstruction(str);
	
			case Append:
				return AppendCPInstruction.parseInstruction(str);
			
			case Variable:
				return VariableCPInstruction.parseInstruction(str);
				
			case Rand:
				return DataGenCPInstruction.parseInstruction(str);

			case StringInit:
				return StringInitCPInstruction.parseInstruction(str);
				
			case FCall:
				return FunctionCallCPInstruction.parseInstruction(str);

			case ParameterizedBuiltin:
				return ParameterizedBuiltinCPInstruction.parseInstruction(str);
			
			case MultiReturnParameterizedBuiltin:
				return MultiReturnParameterizedBuiltinCPInstruction.parseInstruction(str);
		
			case MultiReturnComplexMatrixBuiltin:
				return MultiReturnComplexMatrixBuiltinCPInstruction.parseInstruction(str);
				
			case MultiReturnBuiltin:
				return MultiReturnBuiltinCPInstruction.parseInstruction(str);
				
			case QSort:
				return QuantileSortCPInstruction.parseInstruction(str);
			
			case QPick:
				return QuantilePickCPInstruction.parseInstruction(str);
			
			case MatrixIndexing:
				execType = ExecType.valueOf( str.split(Instruction.OPERAND_DELIM)[0] ); 
				if( execType == ExecType.CP )
					return IndexingCPInstruction.parseInstruction(str);
				else //exectype CP_FILE
					return MatrixIndexingCPFileInstruction.parseInstruction(str);
			
			case Builtin: 
				String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if(parts[0].equals(Opcodes.LOG.toString()) || parts[0].equals(Opcodes.LOGNZ.toString())) {
					if(InstructionUtils.isInteger(parts[3])) // B=log(A), y=log(x)
						// We exploit the fact the number of threads is specified as an integer at parts 3.
						return UnaryCPInstruction.parseInstruction(str);
					else // B=log(A,10), y=log(x,10)
						return BinaryCPInstruction.parseInstruction(str);
				}
				throw new DMLRuntimeException("Invalid Builtin Instruction: " + str );
			
			case MMTSJ:
				return MMTSJCPInstruction.parseInstruction(str);
			
			case PMMJ:
				return PMMJCPInstruction.parseInstruction(str);
			
			case MMChain:
				return MMChainCPInstruction.parseInstruction(str);
			
			case CentralMoment:
				return CentralMomentCPInstruction.parseInstruction(str);
	
			case Covariance:
				return CovarianceCPInstruction.parseInstruction(str);

			case Compression:
				return CompressionCPInstruction.parseInstruction(str);
			
			case DeCompression:
				return DeCompressionCPInstruction.parseInstruction(str);
			
			case QuantizeCompression:
				LOG.debug("Parsing Quantize Compress instruction");
				return CompressionCPInstruction.parseQuantizationFusedInstruction(str);				

			case Local:
				return LocalCPInstruction.parseInstruction(str);

			case SpoofFused:
				return SpoofCPInstruction.parseInstruction(str);
				
			case Sql:
				return SqlCPInstruction.parseInstruction(str);
				
			case Prefetch:
				return PrefetchCPInstruction.parseInstruction(str);
				
			case Broadcast:
				return BroadcastCPInstruction.parseInstruction(str);

			case EvictLineageCache:
				return EvictCPInstruction.parseInstruction(str);

			default:
				throw new DMLRuntimeException("Invalid CP Instruction Type: " + cptype );
		}
	}
}
