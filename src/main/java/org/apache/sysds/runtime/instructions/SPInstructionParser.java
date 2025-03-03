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

import org.apache.sysds.common.InstructionType;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.Compression;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.DeCompression;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.lops.WeightedCrossEntropy;
import org.apache.sysds.lops.WeightedCrossEntropyR;
import org.apache.sysds.lops.WeightedDivMM;
import org.apache.sysds.lops.WeightedDivMMR;
import org.apache.sysds.lops.WeightedSigmoid;
import org.apache.sysds.lops.WeightedSigmoidR;
import org.apache.sysds.lops.WeightedSquaredLoss;
import org.apache.sysds.lops.WeightedSquaredLossR;
import org.apache.sysds.lops.WeightedUnaryMM;
import org.apache.sysds.lops.WeightedUnaryMMR;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.AggregateTernarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateUnarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateUnarySketchSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGAlignedSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendMSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendRSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinUaggChainSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.BuiltinNarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.CSVReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CastSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CentralMomentSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CheckpointSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CompressionSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CovarianceSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CpmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CtableSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CumulativeAggregateSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CumulativeOffsetSPInstruction;
import org.apache.sysds.runtime.instructions.spark.DeCompressionSPInstruction;
import org.apache.sysds.runtime.instructions.spark.DnnSPInstruction;
import org.apache.sysds.runtime.instructions.spark.IndexingSPInstruction;
import org.apache.sysds.runtime.instructions.spark.LIBSVMReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MapmmChainSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MapmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MatrixReshapeSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MultiReturnParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.instructions.spark.PMapmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.instructions.spark.PmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantilePickSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantileSortSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuaternarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.RandSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReorgSPInstruction;
import org.apache.sysds.runtime.instructions.spark.RmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.apache.sysds.runtime.instructions.spark.SpoofSPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.Tsmm2SPInstruction;
import org.apache.sysds.runtime.instructions.spark.TsmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.UaggOuterChainSPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnaryFrameSPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnaryMatrixSPInstruction;
import org.apache.sysds.runtime.instructions.spark.WriteSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ZipmmSPInstruction;


public class SPInstructionParser extends InstructionParser
{

	public static SPInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		InstructionType cptype = InstructionUtils.getSPType(str);
		if ( cptype == null )
			// return null;
			throw new DMLRuntimeException("Invalid SP Instruction Type: " + str);
		SPInstruction spinst = parseSingleInstruction(cptype, str);
		if ( spinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return spinst;
	}

	public static SPInstruction parseSingleInstruction ( InstructionType sptype, String str ) {
		if ( str == null || str.isEmpty() )
			return null;

		String [] parts = null;
		switch(sptype)
		{
			// matrix multiplication instructions
			case CPMM:
				return CpmmSPInstruction.parseInstruction(str);
			case RMM:
				return RmmSPInstruction.parseInstruction(str);
			case MAPMM:
				return MapmmSPInstruction.parseInstruction(str);
			case MAPMMCHAIN:
				return MapmmChainSPInstruction.parseInstruction(str);
			case TSMM:
				return TsmmSPInstruction.parseInstruction(str);
			case TSMM2:
				return Tsmm2SPInstruction.parseInstruction(str);
			case PMM:
				return PmmSPInstruction.parseInstruction(str);
			case ZIPMM:
				return ZipmmSPInstruction.parseInstruction(str);
			case PMAPMM:
				return PMapmmSPInstruction.parseInstruction(str);


			case UaggOuterChain:
				return UaggOuterChainSPInstruction.parseInstruction(str);

			case AggregateUnary:
				return AggregateUnarySPInstruction.parseInstruction(str);

			case AggregateUnarySketch:
				return AggregateUnarySketchSPInstruction.parseInstruction(str);

			case AggregateTernary:
				return AggregateTernarySPInstruction.parseInstruction(str);

			case Dnn:
				 return DnnSPInstruction.parseInstruction(str);

			case MatrixIndexing:
				return IndexingSPInstruction.parseInstruction(str);

			case Reorg:
				return ReorgSPInstruction.parseInstruction(str);

			case Binary:
				return BinarySPInstruction.parseInstruction(str);

			case Ternary:
				return TernarySPInstruction.parseInstruction(str);

			//ternary instructions
			case Ctable:
				return CtableSPInstruction.parseInstruction(str);

			//quaternary instructions
			case Quaternary:
				return QuaternarySPInstruction.parseInstruction(str);

			// Reblock instructions
			case Reblock:
				return ReblockSPInstruction.parseInstruction(str);

			case CSVReblock:
				return CSVReblockSPInstruction.parseInstruction(str);

			case LIBSVMReblock:
				return LIBSVMReblockSPInstruction.parseInstruction(str);

			case Builtin:
				parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if ( parts[0].equals(Opcodes.LOG.toString()) || parts[0].equals(Opcodes.LOGNZ.toString()) ) {
					if ( parts.length == 3 ) {
						// B=log(A), y=log(x)
						return UnaryMatrixSPInstruction.parseInstruction(str);
					} else if ( parts.length == 4 ) {
						// B=log(A,10), y=log(x,10)
						return BinarySPInstruction.parseInstruction(str);
					}
				}
				else {
					throw new DMLRuntimeException("Invalid Builtin Instruction: " + str );
				}

			case Unary:
				parts = InstructionUtils.getInstructionPartsWithValueType(str);
				CPOperand in = new CPOperand(parts[1]);
				if(in.getDataType() == Types.DataType.MATRIX)
					return UnaryMatrixSPInstruction.parseInstruction(str);
				else
					return UnaryFrameSPInstruction.parseInstruction(str);
			case BuiltinNary:
				return BuiltinNarySPInstruction.parseInstruction(str);

			case ParameterizedBuiltin:
				return ParameterizedBuiltinSPInstruction.parseInstruction(str);

			case MultiReturnBuiltin:
				return MultiReturnParameterizedBuiltinSPInstruction.parseInstruction(str);

			case MatrixReshape:
				return MatrixReshapeSPInstruction.parseInstruction(str);

			case MAppend: //matrix/frame
				return AppendMSPInstruction.parseInstruction(str);

			case RAppend: //matrix/frame
				return AppendRSPInstruction.parseInstruction(str);

			case GAppend:
				return AppendGSPInstruction.parseInstruction(str);

			case GAlignedAppend:
				return AppendGAlignedSPInstruction.parseInstruction(str);

			case Rand:
				return RandSPInstruction.parseInstruction(str);

			case QSort:
				return QuantileSortSPInstruction.parseInstruction(str);

			case QPick:
				return QuantilePickSPInstruction.parseInstruction(str);

			case Write:
				return WriteSPInstruction.parseInstruction(str);

			case CumsumAggregate:
				return CumulativeAggregateSPInstruction.parseInstruction(str);

			case CumsumOffset:
				return CumulativeOffsetSPInstruction.parseInstruction(str);

			case CentralMoment:
				return CentralMomentSPInstruction.parseInstruction(str);

			case Covariance:
				return CovarianceSPInstruction.parseInstruction(str);

			case BinUaggChain:
				return BinUaggChainSPInstruction.parseInstruction(str);

			case Checkpoint:
				return CheckpointSPInstruction.parseInstruction(str);

			case Compression:
				return CompressionSPInstruction.parseInstruction(str);

			case DeCompression:
				return DeCompressionSPInstruction.parseInstruction(str);

			case SpoofFused:
				return SpoofSPInstruction.parseInstruction(str);

			case Cast:
				return CastSPInstruction.parseInstruction(str);

			default:
				throw new DMLRuntimeException("Invalid SP Instruction Type: " + sptype );
		}
	}
}
