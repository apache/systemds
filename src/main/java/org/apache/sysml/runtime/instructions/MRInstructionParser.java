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

package org.apache.sysml.runtime.instructions;

import java.util.HashMap;

import org.apache.sysml.lops.BinaryM;
import org.apache.sysml.lops.DataGen;
import org.apache.sysml.lops.MapMult;
import org.apache.sysml.lops.MapMultChain;
import org.apache.sysml.lops.WeightedCrossEntropy;
import org.apache.sysml.lops.WeightedCrossEntropyR;
import org.apache.sysml.lops.WeightedDivMM;
import org.apache.sysml.lops.WeightedDivMMR;
import org.apache.sysml.lops.WeightedSigmoid;
import org.apache.sysml.lops.WeightedSigmoidR;
import org.apache.sysml.lops.WeightedSquaredLoss;
import org.apache.sysml.lops.WeightedSquaredLossR;
import org.apache.sysml.lops.WeightedUnaryMM;
import org.apache.sysml.lops.WeightedUnaryMMR;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.mr.AggregateBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateUnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.AppendInstruction;
import org.apache.sysml.runtime.instructions.mr.BinUaggChainInstruction;
import org.apache.sysml.runtime.instructions.mr.BinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.BinaryMInstruction;
import org.apache.sysml.runtime.instructions.mr.CM_N_COVInstruction;
import org.apache.sysml.runtime.instructions.mr.CSVReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.CSVWriteInstruction;
import org.apache.sysml.runtime.instructions.mr.CombineBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CombineTernaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CombineUnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CumulativeAggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.CumulativeOffsetInstruction;
import org.apache.sysml.runtime.instructions.mr.CumulativeSplitInstruction;
import org.apache.sysml.runtime.instructions.mr.DataGenMRInstruction;
import org.apache.sysml.runtime.instructions.mr.DataPartitionMRInstruction;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateMInstruction;
import org.apache.sysml.runtime.instructions.mr.MMTSJMRInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction;
import org.apache.sysml.runtime.instructions.mr.MapMultChainInstruction;
import org.apache.sysml.runtime.instructions.mr.MatrixReshapeMRInstruction;
import org.apache.sysml.runtime.instructions.mr.PMMJMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ParameterizedBuiltinMRInstruction;
import org.apache.sysml.runtime.instructions.mr.PickByCountInstruction;
import org.apache.sysml.runtime.instructions.mr.PlusMultInstruction;
import org.apache.sysml.runtime.instructions.mr.QuaternaryInstruction;
import org.apache.sysml.runtime.instructions.mr.RandInstruction;
import org.apache.sysml.runtime.instructions.mr.RangeBasedReIndexInstruction;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.RemoveEmptyMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ReorgInstruction;
import org.apache.sysml.runtime.instructions.mr.ReplicateInstruction;
import org.apache.sysml.runtime.instructions.mr.ScalarInstruction;
import org.apache.sysml.runtime.instructions.mr.SeqInstruction;
import org.apache.sysml.runtime.instructions.mr.TernaryInstruction;
import org.apache.sysml.runtime.instructions.mr.UaggOuterChainInstruction;
import org.apache.sysml.runtime.instructions.mr.UnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.ZeroOutInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction.MRINSTRUCTION_TYPE;
import org.apache.sysml.runtime.matrix.SortMR;


public class MRInstructionParser extends InstructionParser 
{	
	static public HashMap<String, MRINSTRUCTION_TYPE> String2MRInstructionType;
	static {
		String2MRInstructionType = new HashMap<String, MRINSTRUCTION_TYPE>();
		
		// AGG Instruction Opcodes 
		String2MRInstructionType.put( "a+"    , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "ak+"   , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "asqk+" , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "a*"    , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "amax"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "amin"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "amean" , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "avar"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "arimax"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "arimin"  , MRINSTRUCTION_TYPE.Aggregate);

		// AGG_BINARY Instruction Opcodes 
		String2MRInstructionType.put( "cpmm" 	, MRINSTRUCTION_TYPE.AggregateBinary);
		String2MRInstructionType.put( "rmm"  	, MRINSTRUCTION_TYPE.AggregateBinary);
		String2MRInstructionType.put( MapMult.OPCODE, MRINSTRUCTION_TYPE.AggregateBinary);
		
		// AGG_UNARY Instruction Opcodes 
		String2MRInstructionType.put( "ua+"   , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uar+"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uac+"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uak+"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uark+" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uack+" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uasqk+" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarsqk+", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacsqk+", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uamean", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarmean",MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacmean",MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uavar",  MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarvar", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacvar", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "ua*"   , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uamax" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uamin" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uatrace" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uaktrace", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarmax"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarimax"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacmax"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarmin"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarimin"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacmin"  , MRINSTRUCTION_TYPE.AggregateUnary);

		// BUILTIN Instruction Opcodes 
		String2MRInstructionType.put( "abs"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sin"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "cos"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "tan"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "asin" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "acos" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "atan" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sign" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sqrt" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "exp"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "log"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "log_nz"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "slog" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "pow"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "round", MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "ceil" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "floor", MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sprop", MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sigmoid", MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sel+", MRINSTRUCTION_TYPE.Unary);
		
		// Specific UNARY Instruction Opcodes
		String2MRInstructionType.put( "tsmm" , MRINSTRUCTION_TYPE.MMTSJ);
		String2MRInstructionType.put( "pmm" , MRINSTRUCTION_TYPE.PMMJ);
		String2MRInstructionType.put( MapMultChain.OPCODE, MRINSTRUCTION_TYPE.MapMultChain);
		String2MRInstructionType.put( "binuaggchain", MRINSTRUCTION_TYPE.BinUaggChain);
		
		// BINARY and SCALAR Instruction Opcodes 
		String2MRInstructionType.put( "+"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "-"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "s-r"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "*"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "/"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "%%"   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "%/%"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "1-*"  , MRINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2MRInstructionType.put( "so"   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "^"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "max"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "min"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( ">"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( ">="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "<"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "<="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "=="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "!="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "^"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "^2"   , MRINSTRUCTION_TYPE.ArithmeticBinary); //special ^ case
		String2MRInstructionType.put( "*2"   , MRINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2MRInstructionType.put( "-nz"  , MRINSTRUCTION_TYPE.ArithmeticBinary); //special - case
		String2MRInstructionType.put( "+*"   , MRINSTRUCTION_TYPE.ArithmeticBinary2); 
		String2MRInstructionType.put( "-*"   , MRINSTRUCTION_TYPE.ArithmeticBinary2); 
		
		String2MRInstructionType.put( "map+"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map-"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map*"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map/"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map%%"   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map%/%"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map1-*"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map^"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "mapmax"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "mapmin"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map>"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map>="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map<"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map<="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map=="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map!="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
	
		String2MRInstructionType.put( "uaggouterchain", MRINSTRUCTION_TYPE.UaggOuterChain);
		
		// REORG Instruction Opcodes 
		String2MRInstructionType.put( "r'"     , MRINSTRUCTION_TYPE.Reorg);
		String2MRInstructionType.put( "rev"     , MRINSTRUCTION_TYPE.Reorg);
		String2MRInstructionType.put( "rdiag"  , MRINSTRUCTION_TYPE.Reorg);
		
		// REPLICATE Instruction Opcodes
		String2MRInstructionType.put( "rep"     , MRINSTRUCTION_TYPE.Replicate);
		
		// DataGen Instruction Opcodes 
		String2MRInstructionType.put( DataGen.RAND_OPCODE   , MRINSTRUCTION_TYPE.Rand);
		String2MRInstructionType.put( DataGen.SEQ_OPCODE   , MRINSTRUCTION_TYPE.Seq);
		
		// REBLOCK Instruction Opcodes 
		String2MRInstructionType.put( "rblk"   , MRINSTRUCTION_TYPE.Reblock);
		String2MRInstructionType.put( "csvrblk", MRINSTRUCTION_TYPE.CSVReblock);
		
		// Ternary Reorg Instruction Opcodes 
		String2MRInstructionType.put( "ctabletransform", MRINSTRUCTION_TYPE.Ternary);
		String2MRInstructionType.put( "ctabletransformscalarweight", MRINSTRUCTION_TYPE.Ternary);
		String2MRInstructionType.put( "ctableexpandscalarweight", MRINSTRUCTION_TYPE.Ternary);
		String2MRInstructionType.put( "ctabletransformhistogram", MRINSTRUCTION_TYPE.Ternary);
		String2MRInstructionType.put( "ctabletransformweightedhistogram", MRINSTRUCTION_TYPE.Ternary);
		
		// Quaternary Instruction Opcodes
		String2MRInstructionType.put( WeightedSquaredLoss.OPCODE,  MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedSquaredLossR.OPCODE, MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedSigmoid.OPCODE,      MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedSigmoidR.OPCODE,     MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedDivMM.OPCODE,        MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedDivMMR.OPCODE,       MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedCrossEntropy.OPCODE, MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedCrossEntropyR.OPCODE,MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedUnaryMM.OPCODE,      MRINSTRUCTION_TYPE.Quaternary);
		String2MRInstructionType.put( WeightedUnaryMMR.OPCODE,     MRINSTRUCTION_TYPE.Quaternary);
		
		// Combine Instruction Opcodes
		String2MRInstructionType.put( "combinebinary" , MRINSTRUCTION_TYPE.CombineBinary);
		String2MRInstructionType.put( "combineunary"  , MRINSTRUCTION_TYPE.CombineUnary);
		String2MRInstructionType.put( "combineternary" , MRINSTRUCTION_TYPE.CombineTernary);
		
		// PickByCount Instruction Opcodes
		String2MRInstructionType.put( "valuepick"  , MRINSTRUCTION_TYPE.PickByCount);  // for quantile()
		String2MRInstructionType.put( "rangepick"  , MRINSTRUCTION_TYPE.PickByCount);  // for interQuantile()
		
		// CM Instruction Opcodes
		String2MRInstructionType.put( "cm"  , MRINSTRUCTION_TYPE.CM_N_COV); 
		String2MRInstructionType.put( "cov"  , MRINSTRUCTION_TYPE.CM_N_COV); 
		String2MRInstructionType.put( "mean"  , MRINSTRUCTION_TYPE.CM_N_COV); 
		
		//groupedAgg Instruction Opcodes
		String2MRInstructionType.put( "groupedagg"  , MRINSTRUCTION_TYPE.GroupedAggregate); 
		String2MRInstructionType.put( "mapgroupedagg"  , MRINSTRUCTION_TYPE.MapGroupedAggregate); 
		
		//rangereindexing
		String2MRInstructionType.put( "rangeReIndex"  , MRINSTRUCTION_TYPE.RangeReIndex);
		String2MRInstructionType.put( "rangeReIndexForLeft"  , MRINSTRUCTION_TYPE.RangeReIndex);
		String2MRInstructionType.put( "zeroOut"  , MRINSTRUCTION_TYPE.ZeroOut);

		//append
		String2MRInstructionType.put( "mappend"  , MRINSTRUCTION_TYPE.Append);
		String2MRInstructionType.put( "rappend"  , MRINSTRUCTION_TYPE.Append);
		String2MRInstructionType.put( "gappend"  , MRINSTRUCTION_TYPE.Append);
		
		//misc
		String2MRInstructionType.put( "rshape", MRINSTRUCTION_TYPE.MatrixReshape);
		
		//partitioning
		String2MRInstructionType.put( "partition", MRINSTRUCTION_TYPE.Partition);
		
		//cumsum/cumprod/cummin/cummax
		String2MRInstructionType.put( "ucumack+"  , MRINSTRUCTION_TYPE.CumsumAggregate);
		String2MRInstructionType.put( "ucumac*"   , MRINSTRUCTION_TYPE.CumsumAggregate);
		String2MRInstructionType.put( "ucumacmin" , MRINSTRUCTION_TYPE.CumsumAggregate);
		String2MRInstructionType.put( "ucumacmax" , MRINSTRUCTION_TYPE.CumsumAggregate);
		String2MRInstructionType.put( "ucumsplit" , MRINSTRUCTION_TYPE.CumsumSplit);
		String2MRInstructionType.put( "bcumoffk+" , MRINSTRUCTION_TYPE.CumsumOffset);
		String2MRInstructionType.put( "bcumoff*"  , MRINSTRUCTION_TYPE.CumsumOffset);
		String2MRInstructionType.put( "bcumoffmin", MRINSTRUCTION_TYPE.CumsumOffset);
		String2MRInstructionType.put( "bcumoffmax", MRINSTRUCTION_TYPE.CumsumOffset);
		
		//dummy (pseudo instructions)
		String2MRInstructionType.put( "sort", MRINSTRUCTION_TYPE.Sort);
		String2MRInstructionType.put( "csvwrite", MRINSTRUCTION_TYPE.CSVWrite);
		
		//parameterized builtins
		String2MRInstructionType.put( "replace", MRINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2MRInstructionType.put( "rexpand", MRINSTRUCTION_TYPE.ParameterizedBuiltin);
		
		//remove empty (special type since binary not unary)
		String2MRInstructionType.put( "rmempty", MRINSTRUCTION_TYPE.RemoveEmpty);
	}
	
	
	public static MRInstruction parseSingleInstruction (String str ) 
		throws DMLRuntimeException 
	{
		if ( str == null || str.isEmpty() )
			return null;
		
		MRINSTRUCTION_TYPE mrtype = InstructionUtils.getMRType(str); 
		return parseSingleInstruction(mrtype, str);
	}
	
	public static MRInstruction parseSingleInstruction (MRINSTRUCTION_TYPE mrtype, String str ) 
		throws DMLRuntimeException 
	{
		if ( str == null || str.isEmpty() )
			return null;
		
		switch(mrtype) 
		{
			case Aggregate:
				return AggregateInstruction.parseInstruction(str);
				
			case ArithmeticBinary: {
				String opcode = InstructionUtils.getOpCode(str);
				String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
				// extract datatypes of first and second input operands
				String dt1 = parts[1].split(Instruction.DATATYPE_PREFIX)[1].split(Instruction.VALUETYPE_PREFIX)[0];
				String dt2 = parts[2].split(Instruction.DATATYPE_PREFIX)[1].split(Instruction.VALUETYPE_PREFIX)[0];
				if ( dt1.equalsIgnoreCase("SCALAR") || dt2.equalsIgnoreCase("SCALAR") ) {
					return ScalarInstruction.parseInstruction(str);
				}
				else {
					if( BinaryM.isOpcode( opcode ) )
						return BinaryMInstruction.parseInstruction(str);
					else
						return BinaryInstruction.parseInstruction(str);
				}
			}
			
			case ArithmeticBinary2:
				return PlusMultInstruction.parseInstruction(str);
			
			case AggregateBinary:
				return AggregateBinaryInstruction.parseInstruction(str);
				
			case AggregateUnary:
				return AggregateUnaryInstruction.parseInstruction(str);
				
			case Ternary: 
				return TernaryInstruction.parseInstruction(str);
			
			case Quaternary: 
				return QuaternaryInstruction.parseInstruction(str);
				
			case Rand:
				return RandInstruction.parseInstruction(str);
				
			case Seq:
				return SeqInstruction.parseInstruction(str);
				
			case Reblock:
				return ReblockInstruction.parseInstruction(str);
			
			case Append:
				return AppendInstruction.parseInstruction(str);
				
			case Reorg:
				return ReorgInstruction.parseInstruction(str);
				
			case Replicate:
				return ReplicateInstruction.parseInstruction(str);
			
			case Unary: {
				String opcode = InstructionUtils.getOpCode(str);
				String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if( parts.length==4 && (opcode.equalsIgnoreCase("log") || opcode.equalsIgnoreCase("log_nz")) )
					return ScalarInstruction.parseInstruction(str);
				else //default case
					return UnaryInstruction.parseInstruction(str);
			}
			case MMTSJ:
				return MMTSJMRInstruction.parseInstruction(str);
			
			case PMMJ:
				return PMMJMRInstruction.parseInstruction(str);
			
			case MapMultChain:
				return MapMultChainInstruction.parseInstruction(str);
			
			case BinUaggChain:
				return BinUaggChainInstruction.parseInstruction(str);
			
			case UaggOuterChain:
				return UaggOuterChainInstruction.parseInstruction(str);
				
			case CombineTernary:
				return CombineTernaryInstruction.parseInstruction(str);
				
			case CombineBinary:
				return CombineBinaryInstruction.parseInstruction(str);
				
			case CombineUnary:
				return CombineUnaryInstruction.parseInstruction(str);
				
			case PickByCount:
				return PickByCountInstruction.parseInstruction(str);
				
			case CM_N_COV:
				return CM_N_COVInstruction.parseInstruction(str);
		
			case GroupedAggregate:
				return GroupedAggregateInstruction.parseInstruction(str);
			
			case MapGroupedAggregate:
				return GroupedAggregateMInstruction.parseInstruction(str);
			
			case RangeReIndex:
				return RangeBasedReIndexInstruction.parseInstruction(str);
			
			case ZeroOut:
				return ZeroOutInstruction.parseInstruction(str);
			
			case MatrixReshape:
				return MatrixReshapeMRInstruction.parseInstruction(str);	
			
			case Sort: //workaround for dummy MR sort instruction
				return SortMR.parseSortInstruction(str);
			
			case CSVReblock:
				return CSVReblockInstruction.parseInstruction(str);
				
			case CSVWrite:
				return CSVWriteInstruction.parseInstruction(str);
				
			case ParameterizedBuiltin:
				return ParameterizedBuiltinMRInstruction.parseInstruction(str);
			
			case RemoveEmpty:
				return RemoveEmptyMRInstruction.parseInstruction(str);
				
			case Partition:
				return DataPartitionMRInstruction.parseInstruction(str);
				
			case CumsumAggregate:
				return CumulativeAggregateInstruction.parseInstruction(str);
				
			case CumsumSplit:
				return CumulativeSplitInstruction.parseInstruction(str);
			
			case CumsumOffset:
				return CumulativeOffsetInstruction.parseInstruction(str);
			
			case INVALID:
			
			default: 
				throw new DMLRuntimeException("Invalid MR Instruction Type: " + mrtype );
		}
	}
	
	public static MRInstruction[] parseMixedInstructions ( String str ) throws DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		Instruction[] inst = InstructionParser.parseMixedInstructions(str);
		MRInstruction[] mrinst = new MRInstruction[inst.length];
		for ( int i=0; i < inst.length; i++ ) {
			mrinst[i] = (MRInstruction) inst[i];
		}
		
		return mrinst;
	}
	
	public static AggregateInstruction[] parseAggregateInstructions(String str) throws DMLRuntimeException 
	{
		AggregateInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new AggregateInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (AggregateInstruction) AggregateInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static ReblockInstruction[] parseReblockInstructions(String str) throws DMLRuntimeException 
	{
		ReblockInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new ReblockInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (ReblockInstruction) ReblockInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static CSVReblockInstruction[] parseCSVReblockInstructions(String str) throws DMLRuntimeException 
	{
		CSVReblockInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new CSVReblockInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (CSVReblockInstruction) CSVReblockInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static CSVWriteInstruction[] parseCSVWriteInstructions(String str) throws DMLRuntimeException 
	{
		CSVWriteInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new CSVWriteInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (CSVWriteInstruction) CSVWriteInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static AggregateBinaryInstruction[] parseAggregateBinaryInstructions(String str) throws DMLRuntimeException 
	{
		AggregateBinaryInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new AggregateBinaryInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (AggregateBinaryInstruction) AggregateBinaryInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static DataGenMRInstruction[] parseDataGenInstructions(String str) throws DMLRuntimeException 
	{
		DataGenMRInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new DataGenMRInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (DataGenMRInstruction) InstructionParser.parseSingleInstruction(strlist[i]);
			}
		}
		return inst;
	}
	
	public static MRInstruction[] parseCombineInstructions(String str) throws DMLRuntimeException 
	{
		MRInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new MRInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				MRINSTRUCTION_TYPE type = InstructionUtils.getMRType(strlist[i]);
				if(type==MRINSTRUCTION_TYPE.CombineBinary)
					inst[i] = (CombineBinaryInstruction) CombineBinaryInstruction.parseInstruction( strlist[i] );
				else if(type==MRINSTRUCTION_TYPE.CombineTernary)
					inst[i] = (CombineTernaryInstruction)CombineTernaryInstruction.parseInstruction(strlist[i]);
				else
					throw new DMLRuntimeException("unknown combine instruction: "+strlist[i]);
			}
		}
		return inst;
	}
	
	public static CM_N_COVInstruction[] parseCM_N_COVInstructions(String str) throws DMLRuntimeException 
	{
		CM_N_COVInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new CM_N_COVInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (CM_N_COVInstruction) CM_N_COVInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}

	public static GroupedAggregateInstruction[] parseGroupedAggInstructions(String str) 
	throws DMLRuntimeException{
		GroupedAggregateInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new GroupedAggregateInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (GroupedAggregateInstruction) GroupedAggregateInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
}
