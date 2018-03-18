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

import org.apache.sysml.lops.Append;
import org.apache.sysml.lops.DataGen;
import org.apache.sysml.lops.LeftIndex;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.RightIndex;
import org.apache.sysml.lops.UnaryCP;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.AppendCPInstruction;
import org.apache.sysml.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.BuiltinNaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysml.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CompressionCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ConvolutionCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysml.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysml.runtime.instructions.cp.DataPartitionCPInstruction;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MatrixReshapeCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysml.runtime.instructions.cp.PMMJCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysml.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.QuantilePickCPInstruction;
import org.apache.sysml.runtime.instructions.cp.QuantileSortCPInstruction;
import org.apache.sysml.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysml.runtime.instructions.cp.SpoofCPInstruction;
import org.apache.sysml.runtime.instructions.cp.StringInitCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysml.runtime.instructions.cp.UaggOuterChainCPInstruction;
import org.apache.sysml.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.instructions.cpfile.MatrixIndexingCPFileInstruction;
import org.apache.sysml.runtime.instructions.cpfile.ParameterizedBuiltinCPFileInstruction;

public class CPInstructionParser extends InstructionParser 
{
	
	public static final HashMap<String, CPType> String2CPInstructionType;
	public static final HashMap<String, CPType> String2CPFileInstructionType;
	
	static {
		String2CPInstructionType = new HashMap<>();
		String2CPInstructionType.put( "ba+*"   	, CPType.AggregateBinary);
		String2CPInstructionType.put( "tak+*"   , CPType.AggregateTernary);
		String2CPInstructionType.put( "tack+*"  , CPType.AggregateTernary);
		
		String2CPInstructionType.put( "uak+"   	, CPType.AggregateUnary);
		String2CPInstructionType.put( "uark+"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uack+"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uasqk+"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarsqk+" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacsqk+" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uamean"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarmean" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacmean" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uavar"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarvar"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacvar"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uamax"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarmax"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarimax", CPType.AggregateUnary);
		String2CPInstructionType.put( "uacmax"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uamin"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarmin"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarimin" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacmin"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "ua+"     , CPType.AggregateUnary);
		String2CPInstructionType.put( "uar+"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "uac+"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "ua*"     , CPType.AggregateUnary);
		String2CPInstructionType.put( "uatrace" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uaktrace", CPType.AggregateUnary);
		String2CPInstructionType.put( "nrow"    ,CPType.AggregateUnary);
		String2CPInstructionType.put( "ncol"    ,CPType.AggregateUnary);
		String2CPInstructionType.put( "length"  ,CPType.AggregateUnary);

		String2CPInstructionType.put( "uaggouterchain", CPType.UaggOuterChain);
		
		// Arithmetic Instruction Opcodes 
		String2CPInstructionType.put( "+"    , CPType.Binary);
		String2CPInstructionType.put( "-"    , CPType.Binary);
		String2CPInstructionType.put( "*"    , CPType.Binary);
		String2CPInstructionType.put( "/"    , CPType.Binary);
		String2CPInstructionType.put( "%%"   , CPType.Binary);
		String2CPInstructionType.put( "%/%"  , CPType.Binary);
		String2CPInstructionType.put( "^"    , CPType.Binary);
		String2CPInstructionType.put( "1-*"  , CPType.Binary); //special * case
		String2CPInstructionType.put( "^2"   , CPType.Binary); //special ^ case
		String2CPInstructionType.put( "*2"   , CPType.Binary); //special * case
		String2CPInstructionType.put( "-nz"  , CPType.Binary); //special - case
		
		// Boolean Instruction Opcodes 
		String2CPInstructionType.put( "&&"   , CPType.Binary);
		String2CPInstructionType.put( "||"   , CPType.Binary);
		String2CPInstructionType.put( "xor"  , CPType.Binary);
		String2CPInstructionType.put( "bitwAnd", CPType.Binary);
		String2CPInstructionType.put( "bitwOr", CPType.Binary);
		String2CPInstructionType.put( "bitwXor", CPType.Binary);
		String2CPInstructionType.put( "bitwShiftL", CPType.Binary);
		String2CPInstructionType.put( "bitwShiftR", CPType.Binary);
		String2CPInstructionType.put( "!"    , CPType.Unary);

		// Relational Instruction Opcodes 
		String2CPInstructionType.put( "=="   , CPType.Binary);
		String2CPInstructionType.put( "!="   , CPType.Binary);
		String2CPInstructionType.put( "<"    , CPType.Binary);
		String2CPInstructionType.put( ">"    , CPType.Binary);
		String2CPInstructionType.put( "<="   , CPType.Binary);
		String2CPInstructionType.put( ">="   , CPType.Binary);
		
		// Builtin Instruction Opcodes 
		String2CPInstructionType.put( "log"  , CPType.Builtin);
		String2CPInstructionType.put( "log_nz"  , CPType.Builtin);

		String2CPInstructionType.put( "max"  , CPType.Binary);
		String2CPInstructionType.put( "min"  , CPType.Binary);
		String2CPInstructionType.put( "solve"  , CPType.Binary);
		
		String2CPInstructionType.put( "exp"   , CPType.Unary);
		String2CPInstructionType.put( "abs"   , CPType.Unary);
		String2CPInstructionType.put( "sin"   , CPType.Unary);
		String2CPInstructionType.put( "cos"   , CPType.Unary);
		String2CPInstructionType.put( "tan"   , CPType.Unary);
		String2CPInstructionType.put( "sinh"   , CPType.Unary);
		String2CPInstructionType.put( "cosh"   , CPType.Unary);
		String2CPInstructionType.put( "tanh"   , CPType.Unary);
		String2CPInstructionType.put( "asin"  , CPType.Unary);
		String2CPInstructionType.put( "acos"  , CPType.Unary);
		String2CPInstructionType.put( "atan"  , CPType.Unary);
		String2CPInstructionType.put( "sign"  , CPType.Unary);
		String2CPInstructionType.put( "sqrt"  , CPType.Unary);
		String2CPInstructionType.put( "plogp" , CPType.Unary);
		String2CPInstructionType.put( "print" , CPType.Unary);
		String2CPInstructionType.put( "assert" , CPType.Unary);
		String2CPInstructionType.put( "round" , CPType.Unary);
		String2CPInstructionType.put( "ceil"  , CPType.Unary);
		String2CPInstructionType.put( "floor" , CPType.Unary);
		String2CPInstructionType.put( "ucumk+", CPType.Unary);
		String2CPInstructionType.put( "ucum*" , CPType.Unary);
		String2CPInstructionType.put( "ucummin", CPType.Unary);
		String2CPInstructionType.put( "ucummax", CPType.Unary);
		String2CPInstructionType.put( "stop"  , CPType.Unary);
		String2CPInstructionType.put( "inverse", CPType.Unary);
		String2CPInstructionType.put( "cholesky",CPType.Unary);
		String2CPInstructionType.put( "sprop", CPType.Unary);
		String2CPInstructionType.put( "sigmoid", CPType.Unary);
		
		String2CPInstructionType.put( "printf" , CPType.BuiltinNary);
		String2CPInstructionType.put( "cbind" , CPType.BuiltinNary);
		String2CPInstructionType.put( "rbind" , CPType.BuiltinNary);
		String2CPInstructionType.put( "eval" , CPType.BuiltinNary);

		// Parameterized Builtin Functions
		String2CPInstructionType.put( "cdf"	 		, CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "invcdf"	 	, CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "groupedagg"	, CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "rmempty"	    , CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "replace"	    , CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "rexpand"	    , CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "toString"    , CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformapply",CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformdecode",CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformcolmap",CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformmeta",CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformencode",CPType.MultiReturnParameterizedBuiltin);
		
		// Ternary Instruction Opcodes
		String2CPInstructionType.put( "+*",      CPType.Ternary);
		String2CPInstructionType.put( "-*",      CPType.Ternary);
		String2CPInstructionType.put( "ifelse",  CPType.Ternary);
		
		// Variable Instruction Opcodes 
		String2CPInstructionType.put( "assignvar"   , CPType.Variable);
		String2CPInstructionType.put( "cpvar"    	, CPType.Variable);
		String2CPInstructionType.put( "mvvar"    	, CPType.Variable);
		String2CPInstructionType.put( "rmvar"    	, CPType.Variable);
		String2CPInstructionType.put( "rmfilevar"   , CPType.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_SCALAR_OPCODE, CPType.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_MATRIX_OPCODE, CPType.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_FRAME_OPCODE,  CPType.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_DOUBLE_OPCODE, CPType.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_INT_OPCODE,    CPType.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_BOOLEAN_OPCODE, CPType.Variable);
		String2CPInstructionType.put( "attachfiletovar"  , CPType.Variable);
		String2CPInstructionType.put( "read"  		, CPType.Variable);
		String2CPInstructionType.put( "write" 		, CPType.Variable);
		String2CPInstructionType.put( "createvar"   , CPType.Variable);

		// Reorg Instruction Opcodes (repositioning of existing values)
		String2CPInstructionType.put( "r'"   	    , CPType.Reorg);
		String2CPInstructionType.put( "rev"   	    , CPType.Reorg);
		String2CPInstructionType.put( "rdiag"       , CPType.Reorg);
		String2CPInstructionType.put( "rshape"      , CPType.MatrixReshape);
		String2CPInstructionType.put( "rsort"      , CPType.Reorg);

		// Opcodes related to convolutions
		String2CPInstructionType.put( "relu_backward"      , CPType.Convolution);
		String2CPInstructionType.put( "relu_maxpooling"      , CPType.Convolution);
		String2CPInstructionType.put( "relu_maxpooling_backward"      , CPType.Convolution);
		String2CPInstructionType.put( "maxpooling"      , CPType.Convolution);
		String2CPInstructionType.put( "maxpooling_backward"      , CPType.Convolution);
		String2CPInstructionType.put( "avgpooling"      , CPType.Convolution);
		String2CPInstructionType.put( "avgpooling_backward"      , CPType.Convolution);
		String2CPInstructionType.put( "conv2d"      , CPType.Convolution);
		String2CPInstructionType.put( "conv2d_bias_add"      , CPType.Convolution);
		String2CPInstructionType.put( "conv2d_backward_filter"      , CPType.Convolution);
		String2CPInstructionType.put( "conv2d_backward_data"      , CPType.Convolution);
		String2CPInstructionType.put( "bias_add"      , CPType.Convolution);
		String2CPInstructionType.put( "bias_multiply"      , CPType.Convolution);
		String2CPInstructionType.put( "channel_sums"      , CPType.Convolution);
		
		// Quaternary instruction opcodes
		String2CPInstructionType.put( "wsloss"  , CPType.Quaternary);
		String2CPInstructionType.put( "wsigmoid", CPType.Quaternary);
		String2CPInstructionType.put( "wdivmm"  , CPType.Quaternary);
		String2CPInstructionType.put( "wcemm"   , CPType.Quaternary);
		String2CPInstructionType.put( "wumm"    , CPType.Quaternary);
		
		// User-defined function Opcodes
		String2CPInstructionType.put( "extfunct"   	, CPType.External);

		String2CPInstructionType.put( Append.OPCODE, CPType.Append);
		
		// data generation opcodes
		String2CPInstructionType.put( DataGen.RAND_OPCODE   , CPType.Rand);
		String2CPInstructionType.put( DataGen.SEQ_OPCODE    , CPType.Rand);
		String2CPInstructionType.put( DataGen.SINIT_OPCODE  , CPType.StringInit);
		String2CPInstructionType.put( DataGen.SAMPLE_OPCODE , CPType.Rand);
		
		String2CPInstructionType.put( "ctable", 		CPType.Ctable);
		String2CPInstructionType.put( "ctableexpand", 	CPType.Ctable);
		
		//central moment, covariance, quantiles (sort/pick)
		String2CPInstructionType.put( "cm"    , CPType.CentralMoment);
		String2CPInstructionType.put( "cov"   , CPType.Covariance);
		String2CPInstructionType.put( "qsort"  , CPType.QSort);
		String2CPInstructionType.put( "qpick"  , CPType.QPick);
		
		
		String2CPInstructionType.put( RightIndex.OPCODE, CPType.MatrixIndexing);
		String2CPInstructionType.put( LeftIndex.OPCODE, CPType.MatrixIndexing);
	
		String2CPInstructionType.put( "tsmm"   , CPType.MMTSJ);
		String2CPInstructionType.put( "pmm"   , CPType.PMMJ);
		String2CPInstructionType.put( "mmchain"   , CPType.MMChain);
		
		String2CPInstructionType.put( "qr",    CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "lu",    CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "eigen", CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "svd", 	 CPType.MultiReturnBuiltin);

		String2CPInstructionType.put( "partition", 	CPType.Partition);
		String2CPInstructionType.put( "compress", 	CPType.Compression);
		String2CPInstructionType.put( "spoof", 		CPType.SpoofFused);

		
		//CP FILE instruction
		String2CPFileInstructionType = new HashMap<>();
		String2CPFileInstructionType.put( "rmempty"	    , CPType.ParameterizedBuiltin);
	}

	public static CPInstruction parseSingleInstruction (String str ) 
		throws DMLRuntimeException 
	{
		if ( str == null || str.isEmpty() )
			return null;

		CPType cptype = InstructionUtils.getCPType(str); 
		if ( cptype == null ) 
			throw new DMLRuntimeException("Unable derive cptype for instruction: " + str);
		CPInstruction cpinst = parseSingleInstruction(cptype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static CPInstruction parseSingleInstruction ( CPType cptype, String str ) 
		throws DMLRuntimeException 
	{
		ExecType execType = null; 
		
		if ( str == null || str.isEmpty() ) 
			return null;
		
		switch(cptype) 
		{
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
				
			case Convolution:
				 return ConvolutionCPInstruction.parseInstruction(str);
				
			case UaggOuterChain:
				return UaggOuterChainCPInstruction.parseInstruction(str);
				
			case MatrixReshape:
				return MatrixReshapeCPInstruction.parseInstruction(str);
	
			case Append:
				return AppendCPInstruction.parseInstruction(str);
			
			case Variable:
				return VariableCPInstruction.parseInstruction(str);
				
			case Rand:
				return DataGenCPInstruction.parseInstruction(str);
				
			case StringInit:
				return StringInitCPInstruction.parseInstruction(str);
				
			case External:
				return FunctionCallCPInstruction.parseInstruction(str);
				
			case ParameterizedBuiltin: 
				execType = ExecType.valueOf( str.split(Instruction.OPERAND_DELIM)[0] ); 
				if( execType == ExecType.CP )
					return ParameterizedBuiltinCPInstruction.parseInstruction(str);
				else //exectype CP_FILE
					return ParameterizedBuiltinCPFileInstruction.parseInstruction(str);
	
			case MultiReturnParameterizedBuiltin:
				return MultiReturnParameterizedBuiltinCPInstruction.parseInstruction(str);
				
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
				String []parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if ( parts[0].equals("log") || parts[0].equals("log_nz") ) {
					if ( parts.length == 3 ) {
						// B=log(A), y=log(x)
						return UnaryCPInstruction.parseInstruction(str);
					} else if ( parts.length == 4 ) {
						// B=log(A,10), y=log(x,10)
						return BinaryCPInstruction.parseInstruction(str);
					}
				}
				else {
					throw new DMLRuntimeException("Invalid Builtin Instruction: " + str );
				}
			case MMTSJ:
				return MMTSJCPInstruction.parseInstruction(str);
			
			case PMMJ:
				return PMMJCPInstruction.parseInstruction(str);
			
			case MMChain:
				return MMChainCPInstruction.parseInstruction(str);
			
			case Partition:
				return DataPartitionCPInstruction.parseInstruction(str);
		
			case CentralMoment:
				return CentralMomentCPInstruction.parseInstruction(str);
	
			case Covariance:
				return CovarianceCPInstruction.parseInstruction(str);
	
			case Compression:
				return (CPInstruction) CompressionCPInstruction.parseInstruction(str);
			
			case SpoofFused:
				return SpoofCPInstruction.parseInstruction(str);
			
			default: 
				throw new DMLRuntimeException("Invalid CP Instruction Type: " + cptype );
		}
	}
}
