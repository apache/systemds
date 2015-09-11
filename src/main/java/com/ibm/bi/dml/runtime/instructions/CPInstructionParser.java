/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions;

import java.util.HashMap;

import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.cp.AggregateBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.AggregateTernaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.AggregateUnaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.AppendCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.ArithmeticBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.BooleanBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.BooleanUnaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.BuiltinBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.BuiltinUnaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CentralMomentCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CovarianceCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.DataGenCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.DataPartitionCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.FileCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.MMChainCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.MMTSJCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.MatrixIndexingCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.MatrixReshapeCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.PMMJCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.QuantilePickCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.QuantileSortCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.QuaternaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.RelationalBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.ReorgCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.StringInitCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.TernaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.cpfile.MatrixIndexingCPFileInstruction;
import com.ibm.bi.dml.runtime.instructions.cpfile.ParameterizedBuiltinCPFileInstruction;

public class CPInstructionParser extends InstructionParser 
{
	
	public static final HashMap<String, CPINSTRUCTION_TYPE> String2CPInstructionType;
	public static final HashMap<String, CPINSTRUCTION_TYPE> String2CPFileInstructionType;
	
	static {
		String2CPInstructionType = new HashMap<String, CPINSTRUCTION_TYPE>();

		String2CPInstructionType.put( "ba+*"   	, CPINSTRUCTION_TYPE.AggregateBinary);
		String2CPInstructionType.put( "tak+*"   	, CPINSTRUCTION_TYPE.AggregateTernary);
		
		String2CPInstructionType.put( "uak+"   	, CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uark+"   , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uack+"   , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uamean"  , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uarmean" , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uacmean" , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uamax"   , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uarmax"  , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uarimax", CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uacmax"  , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uamin"   , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uarmin"  , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uarimin" , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uacmin"  , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "ua+"     , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uar+"    , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uac+"    , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "ua*"     , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uatrace" , CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "uaktrace", CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "nrow"    ,CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "ncol"    ,CPINSTRUCTION_TYPE.AggregateUnary);
		String2CPInstructionType.put( "length"  ,CPINSTRUCTION_TYPE.AggregateUnary);

		// Arithmetic Instruction Opcodes 
		String2CPInstructionType.put( "+"    , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "-"    , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "*"    , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "/"    , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "%%"   , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "%/%"  , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "^"    , CPINSTRUCTION_TYPE.ArithmeticBinary);
		String2CPInstructionType.put( "1-*"  , CPINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2CPInstructionType.put( "^2"   , CPINSTRUCTION_TYPE.ArithmeticBinary); //special ^ case
		String2CPInstructionType.put( "*2"   , CPINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2CPInstructionType.put( "-nz"  , CPINSTRUCTION_TYPE.ArithmeticBinary); //special - case
		
		
		// Boolean Instruction Opcodes 
		String2CPInstructionType.put( "&&"   , CPINSTRUCTION_TYPE.BooleanBinary);
		String2CPInstructionType.put( "||"   , CPINSTRUCTION_TYPE.BooleanBinary);
		
		String2CPInstructionType.put( "!"    , CPINSTRUCTION_TYPE.BooleanUnary);

		// Relational Instruction Opcodes 
		String2CPInstructionType.put( "=="   , CPINSTRUCTION_TYPE.RelationalBinary);
		String2CPInstructionType.put( "!="   , CPINSTRUCTION_TYPE.RelationalBinary);
		String2CPInstructionType.put( "<"    , CPINSTRUCTION_TYPE.RelationalBinary);
		String2CPInstructionType.put( ">"    , CPINSTRUCTION_TYPE.RelationalBinary);
		String2CPInstructionType.put( "<="   , CPINSTRUCTION_TYPE.RelationalBinary);
		String2CPInstructionType.put( ">="   , CPINSTRUCTION_TYPE.RelationalBinary);

		// File Instruction Opcodes 
		String2CPInstructionType.put( "rm"   , CPINSTRUCTION_TYPE.File);
		String2CPInstructionType.put( "mv"   , CPINSTRUCTION_TYPE.File);

		// Builtin Instruction Opcodes 
		String2CPInstructionType.put( "log"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "log_nz"  , CPINSTRUCTION_TYPE.Builtin);

		String2CPInstructionType.put( "max"  , CPINSTRUCTION_TYPE.BuiltinBinary);
		String2CPInstructionType.put( "min"  , CPINSTRUCTION_TYPE.BuiltinBinary);
		String2CPInstructionType.put( "solve"  , CPINSTRUCTION_TYPE.BuiltinBinary);
		
		String2CPInstructionType.put( "exp"   , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "abs"   , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "sin"   , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "cos"   , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "tan"   , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "asin"  , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "acos"  , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "atan"  , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "sqrt"  , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "plogp" , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "print" , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "round" , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "ceil"  , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "floor" , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "ucumk+", CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "ucum*" , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "ucummin", CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "ucummax", CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "stop"  , CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "inverse", CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "sprop", CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "sigmoid", CPINSTRUCTION_TYPE.BuiltinUnary);
		String2CPInstructionType.put( "sel+", CPINSTRUCTION_TYPE.BuiltinUnary);
		
		
		// Parameterized Builtin Functions
		String2CPInstructionType.put( "cdf"	 		, CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2CPInstructionType.put( "invcdf"	 	, CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2CPInstructionType.put( "groupedagg"	, CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2CPInstructionType.put( "rmempty"	    , CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2CPInstructionType.put( "replace"	    , CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2CPInstructionType.put( "rexpand"	    , CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		String2CPInstructionType.put( "transform"	, CPINSTRUCTION_TYPE.ParameterizedBuiltin);

		// Variable Instruction Opcodes 
		String2CPInstructionType.put( "assignvar"   , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "cpvar"    	, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "mvvar"    	, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "rmvar"    	, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "rmfilevar"   , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_SCALAR_OPCODE, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_MATRIX_OPCODE, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_DOUBLE_OPCODE, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_INT_OPCODE,    CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( UnaryCP.CAST_AS_BOOLEAN_OPCODE, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "attachfiletovar"  , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "read"  		, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "write" 		, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "createvar"   , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "seqincr" 	, CPINSTRUCTION_TYPE.Variable); 

		// Reorg Instruction Opcodes (repositioning of existing values)
		String2CPInstructionType.put( "r'"   	    , CPINSTRUCTION_TYPE.Reorg);
		String2CPInstructionType.put( "rdiag"       , CPINSTRUCTION_TYPE.Reorg);
		String2CPInstructionType.put( "rshape"      , CPINSTRUCTION_TYPE.MatrixReshape);
		String2CPInstructionType.put( "rsort"      , CPINSTRUCTION_TYPE.Reorg);
	
		// Quaternary instruction opcodes
		String2CPInstructionType.put( "wsloss"  , CPINSTRUCTION_TYPE.Quaternary);
		String2CPInstructionType.put( "wsigmoid", CPINSTRUCTION_TYPE.Quaternary);
		String2CPInstructionType.put( "wdivmm"  , CPINSTRUCTION_TYPE.Quaternary);
		String2CPInstructionType.put( "wcemm"   , CPINSTRUCTION_TYPE.Quaternary);
		
		// User-defined function Opcodes
		String2CPInstructionType.put( "extfunct"   	, CPINSTRUCTION_TYPE.External);

		String2CPInstructionType.put( "append", CPINSTRUCTION_TYPE.Append);
		
		// data generation opcodes
		String2CPInstructionType.put( DataGen.RAND_OPCODE   , CPINSTRUCTION_TYPE.Rand);
		String2CPInstructionType.put( DataGen.SEQ_OPCODE    , CPINSTRUCTION_TYPE.Rand);
		String2CPInstructionType.put( DataGen.SINIT_OPCODE  , CPINSTRUCTION_TYPE.StringInit);
		String2CPInstructionType.put( DataGen.SAMPLE_OPCODE , CPINSTRUCTION_TYPE.Rand);
		
		String2CPInstructionType.put( "ctable", 		CPINSTRUCTION_TYPE.Ternary);
		String2CPInstructionType.put( "ctableexpand", 	CPINSTRUCTION_TYPE.Ternary);
		
		//central moment, covariance, quantiles (sort/pick)
		String2CPInstructionType.put( "cm"    , CPINSTRUCTION_TYPE.CentralMoment);
		String2CPInstructionType.put( "cov"   , CPINSTRUCTION_TYPE.Covariance);
		String2CPInstructionType.put( "qsort"  , CPINSTRUCTION_TYPE.QSort);
		String2CPInstructionType.put( "qpick"  , CPINSTRUCTION_TYPE.QPick);
		
		
		String2CPInstructionType.put( "rangeReIndex", CPINSTRUCTION_TYPE.MatrixIndexing);
		String2CPInstructionType.put( "leftIndex"   , CPINSTRUCTION_TYPE.MatrixIndexing);
	
		String2CPInstructionType.put( "tsmm"   , CPINSTRUCTION_TYPE.MMTSJ);
		String2CPInstructionType.put( "pmm"   , CPINSTRUCTION_TYPE.PMMJ);
		String2CPInstructionType.put( "mmchain"   , CPINSTRUCTION_TYPE.MMChain);
		
		String2CPInstructionType.put( "qr",    CPINSTRUCTION_TYPE.MultiReturnBuiltin);
		String2CPInstructionType.put( "lu",    CPINSTRUCTION_TYPE.MultiReturnBuiltin);
		String2CPInstructionType.put( "eigen", CPINSTRUCTION_TYPE.MultiReturnBuiltin);
		
		String2CPInstructionType.put( "partition", CPINSTRUCTION_TYPE.Partition);
		
		
		//CP FILE instruction
		String2CPFileInstructionType = new HashMap<String, CPINSTRUCTION_TYPE>();

		String2CPFileInstructionType.put( "rmempty"	    , CPINSTRUCTION_TYPE.ParameterizedBuiltin);
	}

	public static CPInstruction parseSingleInstruction (String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;

		CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
		if ( cptype == null ) 
			throw new DMLRuntimeException("Unable derive cptype for instruction: " + str);
		CPInstruction cpinst = CPInstructionParser.parseSingleInstruction(cptype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static CPInstruction parseSingleInstruction ( CPINSTRUCTION_TYPE cptype, String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		ExecType execType = null; 
		
		if ( str == null || str.isEmpty() ) 
			return null;
		switch(cptype) {
		case AggregateUnary:
			return (CPInstruction) AggregateUnaryCPInstruction.parseInstruction(str);
		
		case AggregateBinary:
			return (CPInstruction) AggregateBinaryCPInstruction.parseInstruction(str);

		case AggregateTernary:
			return (CPInstruction) AggregateTernaryCPInstruction.parseInstruction(str);
			
		case ArithmeticBinary:
			return (CPInstruction) ArithmeticBinaryCPInstruction.parseInstruction(str);
		
		case Ternary:
			return (CPInstruction) TernaryCPInstruction.parseInstruction(str);
		
		case Quaternary:
			return (CPInstruction) QuaternaryCPInstruction.parseInstruction(str);
		
		case BooleanBinary:
			return (CPInstruction) BooleanBinaryCPInstruction.parseInstruction(str);
			
		case BooleanUnary:
			return (CPInstruction) BooleanUnaryCPInstruction.parseInstruction(str);
			
		case BuiltinBinary:
			return (CPInstruction) BuiltinBinaryCPInstruction.parseInstruction(str);
			
		case BuiltinUnary:
			return (CPInstruction) BuiltinUnaryCPInstruction.parseInstruction(str);
			
		case Reorg:
			return (CPInstruction) ReorgCPInstruction.parseInstruction(str);
			
		case MatrixReshape:
			return (CPInstruction) MatrixReshapeCPInstruction.parseInstruction(str);	

		case Append:
			return (CPInstruction) AppendCPInstruction.parseInstruction(str);
			
		case RelationalBinary:
			return (CPInstruction) RelationalBinaryCPInstruction.parseInstruction(str);
			
		case File:
			return (CPInstruction) FileCPInstruction.parseInstruction(str);
			
		case Variable:
			return (CPInstruction) VariableCPInstruction.parseInstruction(str);
			
		case Rand:
			return (CPInstruction) DataGenCPInstruction.parseInstruction(str);
			
		case StringInit:
			return (CPInstruction) StringInitCPInstruction.parseInstruction(str);
			
		case External:
			//return (CPInstruction) ExtBuiltinCPInstruction.parseInstruction(str);
			return (CPInstruction) FunctionCallCPInstruction.parseInstruction(str);
			
		case ParameterizedBuiltin: 
			execType = ExecType.valueOf( str.split(Instruction.OPERAND_DELIM)[0] ); 
			if( execType == ExecType.CP )
				return (CPInstruction) ParameterizedBuiltinCPInstruction.parseInstruction(str);
			else //exectype CP_FILE
				return (CPInstruction) ParameterizedBuiltinCPFileInstruction.parseInstruction(str);

		case MultiReturnBuiltin:
			return (CPInstruction) MultiReturnBuiltinCPInstruction.parseInstruction(str);
			
		case QSort: 
			return (CPInstruction) QuantileSortCPInstruction.parseInstruction(str);
		
		case QPick: 
			return (CPInstruction) QuantilePickCPInstruction.parseInstruction(str);
		
		case MatrixIndexing: 
			execType = ExecType.valueOf( str.split(Instruction.OPERAND_DELIM)[0] ); 
			if( execType == ExecType.CP )
				return (CPInstruction) MatrixIndexingCPInstruction.parseInstruction(str);
			else //exectype CP_FILE
				return (CPInstruction) MatrixIndexingCPFileInstruction.parseInstruction(str);
		
		case Builtin: 
			String []parts = InstructionUtils.getInstructionPartsWithValueType(str);
			if ( parts[0].equals("log") || parts[0].equals("log_nz") ) {
				if ( parts.length == 3 ) {
					// B=log(A), y=log(x)
					return (CPInstruction) BuiltinUnaryCPInstruction.parseInstruction(str);
				} else if ( parts.length == 4 ) {
					// B=log(A,10), y=log(x,10)
					return (CPInstruction) BuiltinBinaryCPInstruction.parseInstruction(str);
				}
			}
			else {
				throw new DMLRuntimeException("Invalid Builtin Instruction: " + str );
			}
		case MMTSJ:
			return (CPInstruction) MMTSJCPInstruction.parseInstruction(str);
		case PMMJ:
			return (CPInstruction) PMMJCPInstruction.parseInstruction(str);
		case MMChain:
			return (CPInstruction) MMChainCPInstruction.parseInstruction(str);
		
		case Partition:
			return (CPInstruction) DataPartitionCPInstruction.parseInstruction(str);	

		case CentralMoment:
			return (CPInstruction) CentralMomentCPInstruction.parseInstruction(str);

		case Covariance:
			return (CPInstruction) CovarianceCPInstruction.parseInstruction(str);
			
		case INVALID:
		default: 
			throw new DMLRuntimeException("Invalid CP Instruction Type: " + cptype );
		}
	}
	
	public static CPInstruction[] parseMixedInstructions ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		Instruction[] inst = InstructionParser.parseMixedInstructions(str);
		CPInstruction[] cpinst = new CPInstruction[inst.length];
		for ( int i=0; i < inst.length; i++ ) {
			cpinst[i] = (CPInstruction) inst[i];
		}
		
		return cpinst;
	}
	
	
}
