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
import org.apache.sysml.lops.RightIndex;
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
import org.apache.sysml.runtime.instructions.mr.TernaryInstruction;
import org.apache.sysml.runtime.instructions.mr.QuaternaryInstruction;
import org.apache.sysml.runtime.instructions.mr.RandInstruction;
import org.apache.sysml.runtime.instructions.mr.RangeBasedReIndexInstruction;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.RemoveEmptyMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ReorgInstruction;
import org.apache.sysml.runtime.instructions.mr.ReplicateInstruction;
import org.apache.sysml.runtime.instructions.mr.ScalarInstruction;
import org.apache.sysml.runtime.instructions.mr.SeqInstruction;
import org.apache.sysml.runtime.instructions.mr.CtableInstruction;
import org.apache.sysml.runtime.instructions.mr.UaggOuterChainInstruction;
import org.apache.sysml.runtime.instructions.mr.UnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.ZeroOutInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction.MRType;
import org.apache.sysml.runtime.matrix.SortMR;


public class MRInstructionParser extends InstructionParser 
{	
	static public HashMap<String, MRType> String2MRInstructionType;
	static {
		String2MRInstructionType = new HashMap<>();
		
		// AGG Instruction Opcodes 
		String2MRInstructionType.put( "a+"    , MRType.Aggregate);
		String2MRInstructionType.put( "ak+"   , MRType.Aggregate);
		String2MRInstructionType.put( "asqk+" , MRType.Aggregate);
		String2MRInstructionType.put( "a*"    , MRType.Aggregate);
		String2MRInstructionType.put( "amax"  , MRType.Aggregate);
		String2MRInstructionType.put( "amin"  , MRType.Aggregate);
		String2MRInstructionType.put( "amean" , MRType.Aggregate);
		String2MRInstructionType.put( "avar"  , MRType.Aggregate);
		String2MRInstructionType.put( "arimax"  , MRType.Aggregate);
		String2MRInstructionType.put( "arimin"  , MRType.Aggregate);

		// AGG_BINARY Instruction Opcodes 
		String2MRInstructionType.put( "cpmm" 	, MRType.AggregateBinary);
		String2MRInstructionType.put( "rmm"  	, MRType.AggregateBinary);
		String2MRInstructionType.put( MapMult.OPCODE, MRType.AggregateBinary);
		
		// AGG_UNARY Instruction Opcodes 
		String2MRInstructionType.put( "ua+"   , MRType.AggregateUnary);
		String2MRInstructionType.put( "uar+"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uac+"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uak+"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uark+" , MRType.AggregateUnary);
		String2MRInstructionType.put( "uack+" , MRType.AggregateUnary);
		String2MRInstructionType.put( "uasqk+" , MRType.AggregateUnary);
		String2MRInstructionType.put( "uarsqk+", MRType.AggregateUnary);
		String2MRInstructionType.put( "uacsqk+", MRType.AggregateUnary);
		String2MRInstructionType.put( "uamean", MRType.AggregateUnary);
		String2MRInstructionType.put( "uarmean",MRType.AggregateUnary);
		String2MRInstructionType.put( "uacmean",MRType.AggregateUnary);
		String2MRInstructionType.put( "uavar",  MRType.AggregateUnary);
		String2MRInstructionType.put( "uarvar", MRType.AggregateUnary);
		String2MRInstructionType.put( "uacvar", MRType.AggregateUnary);
		String2MRInstructionType.put( "ua*"   , MRType.AggregateUnary);
		String2MRInstructionType.put( "uamax" , MRType.AggregateUnary);
		String2MRInstructionType.put( "uamin" , MRType.AggregateUnary);
		String2MRInstructionType.put( "uatrace" , MRType.AggregateUnary);
		String2MRInstructionType.put( "uaktrace", MRType.AggregateUnary);
		String2MRInstructionType.put( "uarmax"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uarimax"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uacmax"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uarmin"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uarimin"  , MRType.AggregateUnary);
		String2MRInstructionType.put( "uacmin"  , MRType.AggregateUnary);

		// BUILTIN Instruction Opcodes 
		String2MRInstructionType.put( "abs"  , MRType.Unary);
		String2MRInstructionType.put( "sin"  , MRType.Unary);
		String2MRInstructionType.put( "cos"  , MRType.Unary);
		String2MRInstructionType.put( "tan"  , MRType.Unary);
		String2MRInstructionType.put( "asin" , MRType.Unary);
		String2MRInstructionType.put( "acos" , MRType.Unary);
		String2MRInstructionType.put( "atan" , MRType.Unary);
		String2MRInstructionType.put( "sign" , MRType.Unary);
		String2MRInstructionType.put( "sqrt" , MRType.Unary);
		String2MRInstructionType.put( "exp"  , MRType.Unary);
		String2MRInstructionType.put( "log"  , MRType.Unary);
		String2MRInstructionType.put( "log_nz"  , MRType.Unary);
		String2MRInstructionType.put( "slog" , MRType.Unary);
		String2MRInstructionType.put( "pow"  , MRType.Unary);
		String2MRInstructionType.put( "round", MRType.Unary);
		String2MRInstructionType.put( "ceil" , MRType.Unary);
		String2MRInstructionType.put( "floor", MRType.Unary);
		String2MRInstructionType.put( "sprop", MRType.Unary);
		String2MRInstructionType.put( "sigmoid", MRType.Unary);
		String2MRInstructionType.put( "!", MRType.Unary);
		
		// Specific UNARY Instruction Opcodes
		String2MRInstructionType.put( "tsmm" , MRType.MMTSJ);
		String2MRInstructionType.put( "pmm" , MRType.PMMJ);
		String2MRInstructionType.put( MapMultChain.OPCODE, MRType.MapMultChain);
		String2MRInstructionType.put( "binuaggchain", MRType.BinUaggChain);
		
		// BINARY and SCALAR Instruction Opcodes 
		String2MRInstructionType.put( "+"    , MRType.Binary);
		String2MRInstructionType.put( "-"    , MRType.Binary);
		String2MRInstructionType.put( "s-r"  , MRType.Binary);
		String2MRInstructionType.put( "*"    , MRType.Binary);
		String2MRInstructionType.put( "/"    , MRType.Binary);
		String2MRInstructionType.put( "%%"   , MRType.Binary);
		String2MRInstructionType.put( "%/%"  , MRType.Binary);
		String2MRInstructionType.put( "1-*"  , MRType.Binary); //special * case
		String2MRInstructionType.put( "so"   , MRType.Binary);
		String2MRInstructionType.put( "^"    , MRType.Binary);
		String2MRInstructionType.put( "max"  , MRType.Binary);
		String2MRInstructionType.put( "min"  , MRType.Binary);
		String2MRInstructionType.put( ">"    , MRType.Binary);
		String2MRInstructionType.put( ">="   , MRType.Binary);
		String2MRInstructionType.put( "<"    , MRType.Binary);
		String2MRInstructionType.put( "<="   , MRType.Binary);
		String2MRInstructionType.put( "=="   , MRType.Binary);
		String2MRInstructionType.put( "!="   , MRType.Binary);
		String2MRInstructionType.put( "^"    , MRType.Binary);
		String2MRInstructionType.put( "^2"   , MRType.Binary); //special ^ case
		String2MRInstructionType.put( "*2"   , MRType.Binary); //special * case
		String2MRInstructionType.put( "-nz"  , MRType.Binary); //special - case
		String2MRInstructionType.put( "&&"   , MRType.Binary);
		String2MRInstructionType.put( "||"   , MRType.Binary);
		String2MRInstructionType.put( "xor"  , MRType.Binary);
		String2MRInstructionType.put( "bitwAnd", MRType.Binary);
		String2MRInstructionType.put( "bitwOr", MRType.Binary);
		String2MRInstructionType.put( "bitwXor", MRType.Binary);
		String2MRInstructionType.put( "bitwShiftL", MRType.Binary);
		String2MRInstructionType.put( "bitwShiftR", MRType.Binary);
		
		String2MRInstructionType.put( "map+"    , MRType.Binary);
		String2MRInstructionType.put( "map-"    , MRType.Binary);
		String2MRInstructionType.put( "map*"    , MRType.Binary);
		String2MRInstructionType.put( "map/"    , MRType.Binary);
		String2MRInstructionType.put( "map%%"   , MRType.Binary);
		String2MRInstructionType.put( "map%/%"  , MRType.Binary);
		String2MRInstructionType.put( "map1-*"  , MRType.Binary);
		String2MRInstructionType.put( "map^"    , MRType.Binary);
		String2MRInstructionType.put( "mapmax"  , MRType.Binary);
		String2MRInstructionType.put( "mapmin"  , MRType.Binary);
		String2MRInstructionType.put( "map>"    , MRType.Binary);
		String2MRInstructionType.put( "map>="   , MRType.Binary);
		String2MRInstructionType.put( "map<"    , MRType.Binary);
		String2MRInstructionType.put( "map<="   , MRType.Binary);
		String2MRInstructionType.put( "map=="   , MRType.Binary);
		String2MRInstructionType.put( "map!="   , MRType.Binary);
		String2MRInstructionType.put( "map&&"   , MRType.Binary);
		String2MRInstructionType.put( "map||"   , MRType.Binary);
		String2MRInstructionType.put( "mapxor"  , MRType.Binary);
		String2MRInstructionType.put( "mapbitwAnd", MRType.Binary);
		String2MRInstructionType.put( "mapbitwOr", MRType.Binary);
		String2MRInstructionType.put( "mapbitwXor", MRType.Binary);
		String2MRInstructionType.put( "mapbitwShiftL", MRType.Binary);
		String2MRInstructionType.put( "mapbitwShiftR", MRType.Binary);
		
		// Ternary Instruction Opcodes
		String2MRInstructionType.put( "+*",     MRType.Ternary); 
		String2MRInstructionType.put( "-*",     MRType.Ternary); 
		String2MRInstructionType.put( "ifelse", MRType.Ternary); 
		
		String2MRInstructionType.put( "uaggouterchain", MRType.UaggOuterChain);
		
		// REORG Instruction Opcodes 
		String2MRInstructionType.put( "r'"     , MRType.Reorg);
		String2MRInstructionType.put( "rev"     , MRType.Reorg);
		String2MRInstructionType.put( "rdiag"  , MRType.Reorg);
		
		// REPLICATE Instruction Opcodes
		String2MRInstructionType.put( "rep"     , MRType.Replicate);
		
		// DataGen Instruction Opcodes 
		String2MRInstructionType.put( DataGen.RAND_OPCODE   , MRType.Rand);
		String2MRInstructionType.put( DataGen.SEQ_OPCODE   , MRType.Seq);
		
		// REBLOCK Instruction Opcodes 
		String2MRInstructionType.put( "rblk"   , MRType.Reblock);
		String2MRInstructionType.put( "csvrblk", MRType.CSVReblock);
		
		// Ternary Reorg Instruction Opcodes 
		String2MRInstructionType.put( "ctabletransform", MRType.Ctable);
		String2MRInstructionType.put( "ctabletransformscalarweight", MRType.Ctable);
		String2MRInstructionType.put( "ctableexpandscalarweight", MRType.Ctable);
		String2MRInstructionType.put( "ctabletransformhistogram", MRType.Ctable);
		String2MRInstructionType.put( "ctabletransformweightedhistogram", MRType.Ctable);
		
		// Quaternary Instruction Opcodes
		String2MRInstructionType.put( WeightedSquaredLoss.OPCODE,  MRType.Quaternary);
		String2MRInstructionType.put( WeightedSquaredLossR.OPCODE, MRType.Quaternary);
		String2MRInstructionType.put( WeightedSigmoid.OPCODE,      MRType.Quaternary);
		String2MRInstructionType.put( WeightedSigmoidR.OPCODE,     MRType.Quaternary);
		String2MRInstructionType.put( WeightedDivMM.OPCODE,        MRType.Quaternary);
		String2MRInstructionType.put( WeightedDivMMR.OPCODE,       MRType.Quaternary);
		String2MRInstructionType.put( WeightedCrossEntropy.OPCODE, MRType.Quaternary);
		String2MRInstructionType.put( WeightedCrossEntropyR.OPCODE,MRType.Quaternary);
		String2MRInstructionType.put( WeightedUnaryMM.OPCODE,      MRType.Quaternary);
		String2MRInstructionType.put( WeightedUnaryMMR.OPCODE,     MRType.Quaternary);
		
		// Combine Instruction Opcodes
		String2MRInstructionType.put( "combinebinary" , MRType.CombineBinary);
		String2MRInstructionType.put( "combineunary"  , MRType.CombineUnary);
		String2MRInstructionType.put( "combineternary" , MRType.CombineTernary);
		
		// PickByCount Instruction Opcodes
		String2MRInstructionType.put( "valuepick"  , MRType.PickByCount);  // for quantile()
		String2MRInstructionType.put( "rangepick"  , MRType.PickByCount);  // for interQuantile()
		
		// CM Instruction Opcodes
		String2MRInstructionType.put( "cm"  , MRType.CM_N_COV); 
		String2MRInstructionType.put( "cov"  , MRType.CM_N_COV); 
		String2MRInstructionType.put( "mean"  , MRType.CM_N_COV); 
		
		//groupedAgg Instruction Opcodes
		String2MRInstructionType.put( "groupedagg"  , MRType.GroupedAggregate); 
		String2MRInstructionType.put( "mapgroupedagg"  , MRType.MapGroupedAggregate); 
		
		//right indexing
		String2MRInstructionType.put( RightIndex.OPCODE , MRType.RightIndex);
		String2MRInstructionType.put( RightIndex.OPCODE+"ForLeft" , MRType.RightIndex);
		String2MRInstructionType.put( "zeroOut" , MRType.ZeroOut);

		//append
		String2MRInstructionType.put( "mappend"  , MRType.Append);
		String2MRInstructionType.put( "rappend"  , MRType.Append);
		String2MRInstructionType.put( "gappend"  , MRType.Append);
		
		//misc
		String2MRInstructionType.put( "rshape", MRType.MatrixReshape);
		
		//partitioning
		String2MRInstructionType.put( "partition", MRType.Partition);
		
		//cumsum/cumprod/cummin/cummax
		String2MRInstructionType.put( "ucumack+"  , MRType.CumsumAggregate);
		String2MRInstructionType.put( "ucumac*"   , MRType.CumsumAggregate);
		String2MRInstructionType.put( "ucumacmin" , MRType.CumsumAggregate);
		String2MRInstructionType.put( "ucumacmax" , MRType.CumsumAggregate);
		String2MRInstructionType.put( "ucumsplit" , MRType.CumsumSplit);
		String2MRInstructionType.put( "bcumoffk+" , MRType.CumsumOffset);
		String2MRInstructionType.put( "bcumoff*"  , MRType.CumsumOffset);
		String2MRInstructionType.put( "bcumoffmin", MRType.CumsumOffset);
		String2MRInstructionType.put( "bcumoffmax", MRType.CumsumOffset);
		
		//dummy (pseudo instructions)
		String2MRInstructionType.put( "sort", MRType.Sort);
		String2MRInstructionType.put( "csvwrite", MRType.CSVWrite);
		
		//parameterized builtins
		String2MRInstructionType.put( "replace", MRType.ParameterizedBuiltin);
		String2MRInstructionType.put( "rexpand", MRType.ParameterizedBuiltin);
		
		//remove empty (special type since binary not unary)
		String2MRInstructionType.put( "rmempty", MRType.RemoveEmpty);
	}
	
	
	public static MRInstruction parseSingleInstruction (String str ) 
		throws DMLRuntimeException 
	{
		if ( str == null || str.isEmpty() )
			return null;
		
		MRType mrtype = InstructionUtils.getMRType(str); 
		return parseSingleInstruction(mrtype, str);
	}
	
	public static MRInstruction parseSingleInstruction (MRType mrtype, String str ) 
		throws DMLRuntimeException 
	{
		if ( str == null || str.isEmpty() )
			return null;
		
		switch(mrtype) 
		{
			case Aggregate:
				return AggregateInstruction.parseInstruction(str);
				
			case Binary: {
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
			
			case Ternary:
				return TernaryInstruction.parseInstruction(str);
			
			case AggregateBinary:
				return AggregateBinaryInstruction.parseInstruction(str);
				
			case AggregateUnary:
				return AggregateUnaryInstruction.parseInstruction(str);
				
			case Ctable: 
				return CtableInstruction.parseInstruction(str);
			
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
			
			case RightIndex:
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
				MRType type = InstructionUtils.getMRType(strlist[i]);
				if(type==MRType.CombineBinary)
					inst[i] = (CombineBinaryInstruction) CombineBinaryInstruction.parseInstruction( strlist[i] );
				else if(type==MRType.CombineTernary)
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
