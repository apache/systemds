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

package org.tugraz.sysds.runtime.instructions;

import java.util.StringTokenizer;

import org.tugraz.sysds.hops.Hop.AggOp;
import org.tugraz.sysds.hops.Hop.Direction;
import org.tugraz.sysds.lops.AppendM;
import org.tugraz.sysds.lops.BinaryM;
import org.tugraz.sysds.lops.GroupedAggregateM;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.lops.MapMult;
import org.tugraz.sysds.lops.MapMultChain;
import org.tugraz.sysds.lops.PMMJ;
import org.tugraz.sysds.lops.UAggOuterChain;
import org.tugraz.sysds.lops.WeightedCrossEntropy;
import org.tugraz.sysds.lops.WeightedCrossEntropyR;
import org.tugraz.sysds.lops.WeightedDivMM;
import org.tugraz.sysds.lops.WeightedDivMMR;
import org.tugraz.sysds.lops.WeightedSigmoid;
import org.tugraz.sysds.lops.WeightedSigmoidR;
import org.tugraz.sysds.lops.WeightedSquaredLoss;
import org.tugraz.sysds.lops.WeightedSquaredLossR;
import org.tugraz.sysds.lops.WeightedUnaryMM;
import org.tugraz.sysds.lops.WeightedUnaryMMR;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.lops.PartialAggregate.CorrectionLocationType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.functionobjects.And;
import org.tugraz.sysds.runtime.functionobjects.BitwAnd;
import org.tugraz.sysds.runtime.functionobjects.BitwOr;
import org.tugraz.sysds.runtime.functionobjects.BitwShiftL;
import org.tugraz.sysds.runtime.functionobjects.BitwShiftR;
import org.tugraz.sysds.runtime.functionobjects.BitwXor;
import org.tugraz.sysds.runtime.functionobjects.Builtin;
import org.tugraz.sysds.runtime.functionobjects.CM;
import org.tugraz.sysds.runtime.functionobjects.Divide;
import org.tugraz.sysds.runtime.functionobjects.Equals;
import org.tugraz.sysds.runtime.functionobjects.GreaterThan;
import org.tugraz.sysds.runtime.functionobjects.GreaterThanEquals;
import org.tugraz.sysds.runtime.functionobjects.IfElse;
import org.tugraz.sysds.runtime.functionobjects.IndexFunction;
import org.tugraz.sysds.runtime.functionobjects.IntegerDivide;
import org.tugraz.sysds.runtime.functionobjects.KahanPlus;
import org.tugraz.sysds.runtime.functionobjects.KahanPlusSq;
import org.tugraz.sysds.runtime.functionobjects.LessThan;
import org.tugraz.sysds.runtime.functionobjects.LessThanEquals;
import org.tugraz.sysds.runtime.functionobjects.Mean;
import org.tugraz.sysds.runtime.functionobjects.Minus;
import org.tugraz.sysds.runtime.functionobjects.Minus1Multiply;
import org.tugraz.sysds.runtime.functionobjects.MinusMultiply;
import org.tugraz.sysds.runtime.functionobjects.MinusNz;
import org.tugraz.sysds.runtime.functionobjects.Modulus;
import org.tugraz.sysds.runtime.functionobjects.Multiply;
import org.tugraz.sysds.runtime.functionobjects.Multiply2;
import org.tugraz.sysds.runtime.functionobjects.Not;
import org.tugraz.sysds.runtime.functionobjects.NotEquals;
import org.tugraz.sysds.runtime.functionobjects.Or;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.functionobjects.PlusMultiply;
import org.tugraz.sysds.runtime.functionobjects.Power;
import org.tugraz.sysds.runtime.functionobjects.Power2;
import org.tugraz.sysds.runtime.functionobjects.ReduceAll;
import org.tugraz.sysds.runtime.functionobjects.ReduceCol;
import org.tugraz.sysds.runtime.functionobjects.ReduceDiag;
import org.tugraz.sysds.runtime.functionobjects.ReduceRow;
import org.tugraz.sysds.runtime.functionobjects.Xor;
import org.tugraz.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.tugraz.sysds.runtime.instructions.gpu.GPUInstruction.GPUINSTRUCTION_TYPE;
import org.tugraz.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.tugraz.sysds.runtime.matrix.data.LibCommonsMath;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.RightScalarOperator;
import org.tugraz.sysds.runtime.matrix.operators.ScalarOperator;
import org.tugraz.sysds.runtime.matrix.operators.TernaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.UnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.CMOperator;
import org.tugraz.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


public class InstructionUtils 
{

	public static int checkNumFields( String str, int expected ) {
		//note: split required for empty tokens
		int numParts = str.split(Instruction.OPERAND_DELIM).length;
		int numFields = numParts - 2; // -2 accounts for execType and opcode
		
		if ( numFields != expected ) 
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected + ") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}

	public static int checkNumFields( String[] parts, int expected ) {
		int numParts = parts.length;
		int numFields = numParts - 1; //account for opcode
		
		if ( numFields != expected ) 
			throw new DMLRuntimeException("checkNumFields() -- expected number (" + expected + ") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}

	public static int checkNumFields( String[] parts, int expected1, int expected2 ) {
		int numParts = parts.length;
		int numFields = numParts - 1; //account for opcode
		
		if ( numFields != expected1 && numFields != expected2 ) 
			throw new DMLRuntimeException("checkNumFields() -- expected number (" + expected1 + " or "+ expected2 +") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}

	public static int checkNumFields( String str, int expected1, int expected2 ) {
		//note: split required for empty tokens
		int numParts = str.split(Instruction.OPERAND_DELIM).length;
		int numFields = numParts - 2; // -2 accounts for execType and opcode
		if ( numFields != expected1 && numFields != expected2 ) 
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected1 + " or "+ expected2 +") != is not equal to actual number (" + numFields + ").");
		return numFields; 
	}
	
	/**
	 * Given an instruction string, strip-off the execution type and return 
	 * opcode and all input/output operands WITHOUT their data/value type. 
	 * i.e., ret.length = parts.length-1 (-1 for execution type)
	 * 
	 * @param str instruction string
	 * @return instruction parts as string array
	 */
	public static String[] getInstructionParts( String str ) {
		StringTokenizer st = new StringTokenizer( str, Instruction.OPERAND_DELIM );
		String[] ret = new String[st.countTokens()-1];
		st.nextToken(); // stripping-off the exectype
		ret[0] = st.nextToken(); // opcode
		int index = 1;
		while( st.hasMoreTokens() ){
			String tmp = st.nextToken();
			int ix = tmp.indexOf(Instruction.DATATYPE_PREFIX);
			ret[index++] = tmp.substring(0,((ix>=0)?ix:tmp.length()));	
		}
		return ret;
	}
	
	/**
	 * Given an instruction string, this function strips-off the 
	 * execution type (CP or MR) and returns the remaining parts, 
	 * which include the opcode as well as the input and output operands.
	 * Each returned part will have the datatype and valuetype associated
	 * with the operand.
	 * 
	 * This function is invoked mainly for parsing CPInstructions.
	 * 
	 * @param str instruction string
	 * @return instruction parts as string array
	 */
	public static String[] getInstructionPartsWithValueType( String str ) {
		//note: split required for empty tokens
		String[] parts = str.split(Instruction.OPERAND_DELIM, -1);
		String[] ret = new String[parts.length-1]; // stripping-off the exectype
		ret[0] = parts[1]; // opcode
		for( int i=1; i<parts.length; i++ )
			ret[i-1] = parts[i];
		
		return ret;
	}
	
	public static ExecType getExecType( String str ) {
		int ix = str.indexOf(Instruction.OPERAND_DELIM);
		return ExecType.valueOf(str.substring(0, ix));
	}

	public static String getOpCode( String str ) {
		int ix1 = str.indexOf(Instruction.OPERAND_DELIM);
		int ix2 = str.indexOf(Instruction.OPERAND_DELIM, ix1+1);
		return str.substring(ix1+1, ix2);
	}

	public static SPType getSPType(String str) {
		return SPInstructionParser.String2SPInstructionType.get(getOpCode(str));
	}

	public static CPType getCPType(String str) {
		return CPInstructionParser.String2CPInstructionType.get(getOpCode(str));
	}
	
	public static SPType getSPTypeByOpcode(String opcode) {
		return SPInstructionParser.String2SPInstructionType.get(opcode);
	}
	
	public static CPType getCPTypeByOpcode( String opcode ) {
		return CPInstructionParser.String2CPInstructionType.get(opcode);
	}

	public static GPUINSTRUCTION_TYPE getGPUType( String str ) {
		return GPUInstructionParser.String2GPUInstructionType.get(getOpCode(str));
	}

	public static boolean isBuiltinFunction( String opcode ) {
		Builtin.BuiltinCode bfc = Builtin.String2BuiltinCode.get(opcode);
		return (bfc != null);
	}

	/**
	 * Evaluates if at least one instruction of the given instruction set
	 * used the distributed cache; this call can also be used for individual
	 * instructions. 
	 * 
	 * @param str instruction set
	 * @return true if at least one instruction uses distributed cache
	 */
	public static boolean isDistributedCacheUsed(String str) 
	{	
		String[] parts = str.split(Instruction.INSTRUCTION_DELIM);
		for(String inst : parts) 
		{
			String opcode = getOpCode(inst);
			if(  opcode.equalsIgnoreCase(AppendM.OPCODE)  
			   || opcode.equalsIgnoreCase(MapMult.OPCODE)
			   || opcode.equalsIgnoreCase(MapMultChain.OPCODE)
			   || opcode.equalsIgnoreCase(PMMJ.OPCODE)
			   || opcode.equalsIgnoreCase(UAggOuterChain.OPCODE)
			   || opcode.equalsIgnoreCase(GroupedAggregateM.OPCODE)
			   || isDistQuaternaryOpcode( opcode ) //multiple quaternary opcodes
			   || BinaryM.isOpcode( opcode ) ) //multiple binary opcodes	
			{
				return true;
			}
		}
		return false;
	}

	public static AggregateUnaryOperator parseBasicAggregateUnaryOperator(String opcode) {
		return parseBasicAggregateUnaryOperator(opcode, 1);
	}
	
	public static AggregateUnaryOperator parseBasicAggregateUnaryOperator(String opcode, int numThreads)
	{
		AggregateUnaryOperator aggun = null;
		
		if ( opcode.equalsIgnoreCase("uak+") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uark+") ) { // RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uack+") ) { // ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTROW);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uasqk+") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uarsqk+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uacsqk+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), true, CorrectionLocationType.LASTROW);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uamean") ) {
			// Mean
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uarmean") ) {
			// RowMeans
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uacmean") ) {
			// ColMeans
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), true, CorrectionLocationType.LASTTWOROWS);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uavar") ) {
			// Variance
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			CorrectionLocationType cloc = CorrectionLocationType.LASTFOURCOLUMNS;
			AggregateOperator agg = new AggregateOperator(0, varFn, true, cloc);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uarvar") ) {
			// RowVariances
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			CorrectionLocationType cloc = CorrectionLocationType.LASTFOURCOLUMNS;
			AggregateOperator agg = new AggregateOperator(0, varFn, true, cloc);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uacvar") ) {
			// ColVariances
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			CorrectionLocationType cloc = CorrectionLocationType.LASTFOURROWS;
			AggregateOperator agg = new AggregateOperator(0, varFn, true, cloc);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("ua+") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uar+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uac+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("ua*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uar*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uac*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uamax") ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uamin") ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uatrace") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uaktrace") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject(), numThreads);
		} 		
		else if ( opcode.equalsIgnoreCase("uarmax") ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if (opcode.equalsIgnoreCase("uarimax") ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uarmin") ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if (opcode.equalsIgnoreCase("uarimin") ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("minindex"), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase("uacmax") ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase("uacmin") ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		
		return aggun;
	}

	public static AggregateTernaryOperator parseAggregateTernaryOperator(String opcode) {
		return parseAggregateTernaryOperator(opcode, 1);
	}
	
	public static AggregateTernaryOperator parseAggregateTernaryOperator(String opcode, int numThreads) {
		CorrectionLocationType corr = opcode.equalsIgnoreCase("tak+*") ? 
				CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.LASTROW;
		AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, corr);
		IndexFunction ixfun = opcode.equalsIgnoreCase("tak+*") ? 
			ReduceAll.getReduceAllFnObject() : ReduceRow.getReduceRowFnObject();
		
		return new AggregateTernaryOperator(Multiply.getMultiplyFnObject(), agg, ixfun, numThreads);
	}
	
	public static AggregateOperator parseAggregateOperator(String opcode, String corrExists, String corrLoc)
	{
		AggregateOperator agg = null;
	
		if ( opcode.equalsIgnoreCase("ak+") || opcode.equalsIgnoreCase("aktrace") ) {
			boolean lcorrExists = (corrExists==null) ? true : Boolean.parseBoolean(corrExists);
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), lcorrExists, lcorrLoc);
		}
		else if ( opcode.equalsIgnoreCase("asqk+") ) {
			boolean lcorrExists = (corrExists==null) ? true : Boolean.parseBoolean(corrExists);
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), lcorrExists, lcorrLoc);
		}
		else if ( opcode.equalsIgnoreCase("a+") ) {
			agg = new AggregateOperator(0, Plus.getPlusFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("a*") ) {
			agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
		}
		else if (opcode.equalsIgnoreCase("arimax")){
			agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
		}
		else if ( opcode.equalsIgnoreCase("amax") ) {
			agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
		}
		else if ( opcode.equalsIgnoreCase("amin") ) {
			agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
		}
		else if (opcode.equalsIgnoreCase("arimin")){
			agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("minindex"), true, CorrectionLocationType.LASTCOLUMN);
		}
		else if ( opcode.equalsIgnoreCase("amean") ) {
			boolean lcorrExists = (corrExists==null) ? true : Boolean.parseBoolean(corrExists);
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTTWOCOLUMNS : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), lcorrExists, lcorrLoc);
		}
		else if ( opcode.equalsIgnoreCase("avar") ) {
			boolean lcorrExists = (corrExists==null) ? true : Boolean.parseBoolean(corrExists);
			CorrectionLocationType lcorrLoc = (corrLoc==null) ?
					CorrectionLocationType.LASTFOURCOLUMNS :
					CorrectionLocationType.valueOf(corrLoc);
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			agg = new AggregateOperator(0, varFn, lcorrExists, lcorrLoc);
		}

		return agg;
	}

	public static AggregateUnaryOperator parseBasicCumulativeAggregateUnaryOperator(UnaryOperator uop) {
		Builtin f = (Builtin)uop.fn;
		if( f.getBuiltinCode()==BuiltinCode.CUMSUM ) 
			return parseBasicAggregateUnaryOperator("uack+") ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMPROD ) 
			return parseBasicAggregateUnaryOperator("uac*") ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMMIN ) 
			return parseBasicAggregateUnaryOperator("uacmin") ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMMAX ) 
			return parseBasicAggregateUnaryOperator("uacmax" ) ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMSUMPROD ) 
			return parseBasicAggregateUnaryOperator("uack+*" ) ;
		throw new RuntimeException("Unsupported cumulative aggregate unary operator: "+f.getBuiltinCode());
	}

	public static AggregateUnaryOperator parseCumulativeAggregateUnaryOperator(String opcode) {
		AggregateOperator agg = null;
		if( "ucumack+".equals(opcode) )
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTROW);
		else if ( "ucumac*".equals(opcode) )
			agg = new AggregateOperator(1, Multiply.getMultiplyFnObject(), false, CorrectionLocationType.NONE);
		else if ( "ucumac+*".equals(opcode) )
			agg = new AggregateOperator(0, PlusMultiply.getFnObject(), false, CorrectionLocationType.NONE);
		else if ( "ucumacmin".equals(opcode) )
			agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"), false, CorrectionLocationType.NONE);
		else if ( "ucumacmax".equals(opcode) )
			agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"), false, CorrectionLocationType.NONE);
		return new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
	}
	
	public static UnaryOperator parseUnaryOperator(String opcode) {
		return opcode.equals("!") ?
			new UnaryOperator(Not.getNotFnObject()) :
			new UnaryOperator(Builtin.getBuiltinFnObject(opcode));
	}

	public static Operator parseBinaryOrBuiltinOperator(String opcode, CPOperand in1, CPOperand in2) {
		if( LibCommonsMath.isSupportedMatrixMatrixOperation(opcode) )
			return null;
		boolean matrixScalar = (in1.getDataType() != in2.getDataType());
		return Builtin.isBuiltinFnObject(opcode) ?
			(matrixScalar ? new RightScalarOperator( Builtin.getBuiltinFnObject(opcode), 0) :
				new BinaryOperator( Builtin.getBuiltinFnObject(opcode))) :
			(matrixScalar ? parseScalarBinaryOperator(opcode, in1.getDataType().isScalar()) :
				parseBinaryOperator(opcode));
	}
	
	public static Operator parseExtendedBinaryOrBuiltinOperator(String opcode, CPOperand in1, CPOperand in2) {
		boolean matrixScalar = (in1.getDataType() != in2.getDataType());
		return Builtin.isBuiltinFnObject(opcode) ?
			(matrixScalar ? new RightScalarOperator( Builtin.getBuiltinFnObject(opcode), 0) :
				new BinaryOperator( Builtin.getBuiltinFnObject(opcode))) :
			(matrixScalar ? parseScalarBinaryOperator(opcode, in1.getDataType().isScalar()) :
				parseExtendedBinaryOperator(opcode));
	}
	
	public static BinaryOperator parseBinaryOperator(String opcode) 
	{
		if(opcode.equalsIgnoreCase("=="))
			return new BinaryOperator(Equals.getEqualsFnObject());
		else if(opcode.equalsIgnoreCase("!="))
			return new BinaryOperator(NotEquals.getNotEqualsFnObject());
		else if(opcode.equalsIgnoreCase("<"))
			return new BinaryOperator(LessThan.getLessThanFnObject());
		else if(opcode.equalsIgnoreCase(">"))
			return new BinaryOperator(GreaterThan.getGreaterThanFnObject());
		else if(opcode.equalsIgnoreCase("<="))
			return new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(">="))
			return new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase("&&"))
			return new BinaryOperator(And.getAndFnObject());
		else if(opcode.equalsIgnoreCase("||"))
			return new BinaryOperator(Or.getOrFnObject());
		else if(opcode.equalsIgnoreCase("xor"))
			return new BinaryOperator(Xor.getXorFnObject());
		else if(opcode.equalsIgnoreCase("bitwAnd"))
			return new BinaryOperator(BitwAnd.getBitwAndFnObject());
		else if(opcode.equalsIgnoreCase("bitwOr"))
			return new BinaryOperator(BitwOr.getBitwOrFnObject());
		else if(opcode.equalsIgnoreCase("bitwXor"))
			return new BinaryOperator(BitwXor.getBitwXorFnObject());
		else if(opcode.equalsIgnoreCase("bitwShiftL"))
			return new BinaryOperator(BitwShiftL.getBitwShiftLFnObject());
		else if(opcode.equalsIgnoreCase("bitwShiftR"))
			return new BinaryOperator(BitwShiftR.getBitwShiftRFnObject());
		else if(opcode.equalsIgnoreCase("+"))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase("-"))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase("*"))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if(opcode.equalsIgnoreCase("1-*"))
			return new BinaryOperator(Minus1Multiply.getMinus1MultiplyFnObject());
		else if ( opcode.equalsIgnoreCase("*2") ) 
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase("/"))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase("%%"))
			return new BinaryOperator(Modulus.getFnObject());
		else if(opcode.equalsIgnoreCase("%/%"))
			return new BinaryOperator(IntegerDivide.getFnObject());
		else if(opcode.equalsIgnoreCase("^"))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase("^2") )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase("max") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase("min") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		
		throw new RuntimeException("Unknown binary opcode " + opcode);
	}
	
	public static TernaryOperator parseTernaryOperator(String opcode) {
		return new TernaryOperator(opcode.equals("+*") ? PlusMultiply.getFnObject() :
			opcode.equals("-*") ? MinusMultiply.getFnObject() : IfElse.getFnObject());
	}
	
	/**
	 * scalar-matrix operator
	 * 
	 * @param opcode the opcode
	 * @param arg1IsScalar ?
	 * @return scalar operator
	 */
	public static ScalarOperator parseScalarBinaryOperator(String opcode, boolean arg1IsScalar) 
	{
		//for all runtimes that set constant dynamically (cp/spark)
		double default_constant = 0;
		
		return parseScalarBinaryOperator(opcode, arg1IsScalar, default_constant);
	}
	
	/**
	 * scalar-matrix operator
	 * 
	 * @param opcode the opcode
	 * @param arg1IsScalar ?
	 * @param constant ?
	 * @return scalar operator
	 */
	public static ScalarOperator parseScalarBinaryOperator(String opcode, boolean arg1IsScalar, double constant)
	{
		//commutative operators
		if ( opcode.equalsIgnoreCase("+") ){ 
			return new RightScalarOperator(Plus.getPlusFnObject(), constant); 
		}
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new RightScalarOperator(Multiply.getMultiplyFnObject(), constant);
		} 
		//non-commutative operators
		else if ( opcode.equalsIgnoreCase("-") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Minus.getMinusFnObject(), constant);
			else return new RightScalarOperator(Minus.getMinusFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("-nz") ) {
			//no support for left scalar yet
			return new RightScalarOperator(MinusNz.getMinusNzFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("/") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Divide.getDivideFnObject(), constant);
			else return new RightScalarOperator(Divide.getDivideFnObject(), constant);
		}  
		else if ( opcode.equalsIgnoreCase("%%") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Modulus.getFnObject(), constant);
			else return new RightScalarOperator(Modulus.getFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("%/%") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(IntegerDivide.getFnObject(), constant);
			else return new RightScalarOperator(IntegerDivide.getFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("^") ){
			if(arg1IsScalar)
				return new LeftScalarOperator(Power.getPowerFnObject(), constant);
			else return new RightScalarOperator(Power.getPowerFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("max"), constant);
		}
		else if ( opcode.equalsIgnoreCase("min") ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("min"), constant);
		}
		else if ( opcode.equalsIgnoreCase("log") || opcode.equalsIgnoreCase("log_nz") ){
			if( arg1IsScalar )
				return new LeftScalarOperator(Builtin.getBuiltinFnObject(opcode), constant);
			return new RightScalarOperator(Builtin.getBuiltinFnObject(opcode), constant);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(GreaterThan.getGreaterThanFnObject(), constant);
			return new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), constant);
			return new RightScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(LessThan.getLessThanFnObject(), constant);
			return new RightScalarOperator(LessThan.getLessThanFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), constant);
			return new RightScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Equals.getEqualsFnObject(), constant);
			return new RightScalarOperator(Equals.getEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(NotEquals.getNotEqualsFnObject(), constant);
			return new RightScalarOperator(NotEquals.getNotEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("&&") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(And.getAndFnObject(), constant) :
				new RightScalarOperator(And.getAndFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("||") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(Or.getOrFnObject(), constant) :
				new RightScalarOperator(Or.getOrFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("xor") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(Xor.getXorFnObject(), constant) :
				new RightScalarOperator(Xor.getXorFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("bitwAnd") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwAnd.getBitwAndFnObject(), constant) :
				new RightScalarOperator(BitwAnd.getBitwAndFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("bitwOr") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwOr.getBitwOrFnObject(), constant) :
				new RightScalarOperator(BitwOr.getBitwOrFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("bitwXor") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwXor.getBitwXorFnObject(), constant) :
				new RightScalarOperator(BitwXor.getBitwXorFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("bitwShiftL") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwShiftL.getBitwShiftLFnObject(), constant) :
				new RightScalarOperator(BitwShiftL.getBitwShiftLFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("bitwShiftR") ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwShiftR.getBitwShiftRFnObject(), constant) :
				new RightScalarOperator(BitwShiftR.getBitwShiftRFnObject(), constant);
		}
		//operations that only exist for performance purposes (all unary or commutative operators)
		else if ( opcode.equalsIgnoreCase("*2") ) {
			return new RightScalarOperator(Multiply2.getMultiply2FnObject(), constant);
		} 
		else if ( opcode.equalsIgnoreCase("^2") ){
			return new RightScalarOperator(Power2.getPower2FnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase("1-*") ) {
			return new RightScalarOperator(Minus1Multiply.getMinus1MultiplyFnObject(), constant);
		}
		
		//operations that only exist in mr
		else if ( opcode.equalsIgnoreCase("s-r") ) {
			return new LeftScalarOperator(Minus.getMinusFnObject(), constant);
		} 
		else if ( opcode.equalsIgnoreCase("so") ) {
			return new LeftScalarOperator(Divide.getDivideFnObject(), constant);
		}
		
		throw new RuntimeException("Unknown binary opcode " + opcode);
	}

	public static BinaryOperator parseExtendedBinaryOperator(String opcode) {
		if(opcode.equalsIgnoreCase("==") || opcode.equalsIgnoreCase("map=="))
			return new BinaryOperator(Equals.getEqualsFnObject());
		else if(opcode.equalsIgnoreCase("!=") || opcode.equalsIgnoreCase("map!="))
			return new BinaryOperator(NotEquals.getNotEqualsFnObject());
		else if(opcode.equalsIgnoreCase("<") || opcode.equalsIgnoreCase("map<"))
			return new BinaryOperator(LessThan.getLessThanFnObject());
		else if(opcode.equalsIgnoreCase(">") || opcode.equalsIgnoreCase("map>"))
			return new BinaryOperator(GreaterThan.getGreaterThanFnObject());
		else if(opcode.equalsIgnoreCase("<=") || opcode.equalsIgnoreCase("map<="))
			return new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(">=") || opcode.equalsIgnoreCase("map>="))
			return new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase("&&") || opcode.equalsIgnoreCase("map&&"))
			return new BinaryOperator(And.getAndFnObject());
		else if(opcode.equalsIgnoreCase("||") || opcode.equalsIgnoreCase("map||"))
			return new BinaryOperator(Or.getOrFnObject());
		else if(opcode.equalsIgnoreCase("xor") || opcode.equalsIgnoreCase("mapxor"))
			return new BinaryOperator(Xor.getXorFnObject());
		else if(opcode.equalsIgnoreCase("bitwAnd") || opcode.equalsIgnoreCase("mapbitwAnd"))
			return new BinaryOperator(BitwAnd.getBitwAndFnObject());
		else if(opcode.equalsIgnoreCase("bitwOr") || opcode.equalsIgnoreCase("mapbitwOr"))
			return new BinaryOperator(BitwOr.getBitwOrFnObject());
		else if(opcode.equalsIgnoreCase("bitwXor") || opcode.equalsIgnoreCase("mapbitwXor"))
			return new BinaryOperator(BitwXor.getBitwXorFnObject());
		else if(opcode.equalsIgnoreCase("bitwShiftL") || opcode.equalsIgnoreCase("mapbitwShiftL"))
			return new BinaryOperator(BitwShiftL.getBitwShiftLFnObject());
		else if(opcode.equalsIgnoreCase("bitwShiftR") || opcode.equalsIgnoreCase("mapbitwShiftR"))
			return new BinaryOperator(BitwShiftR.getBitwShiftRFnObject());
		else if(opcode.equalsIgnoreCase("+") || opcode.equalsIgnoreCase("map+"))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase("-") || opcode.equalsIgnoreCase("map-"))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase("*") || opcode.equalsIgnoreCase("map*"))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if(opcode.equalsIgnoreCase("1-*") || opcode.equalsIgnoreCase("map1-*"))
			return new BinaryOperator(Minus1Multiply.getMinus1MultiplyFnObject());
		else if ( opcode.equalsIgnoreCase("*2") ) 
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase("/") || opcode.equalsIgnoreCase("map/"))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase("%%") || opcode.equalsIgnoreCase("map%%"))
			return new BinaryOperator(Modulus.getFnObject());
		else if(opcode.equalsIgnoreCase("%/%") || opcode.equalsIgnoreCase("map%/%"))
			return new BinaryOperator(IntegerDivide.getFnObject());
		else if(opcode.equalsIgnoreCase("^") || opcode.equalsIgnoreCase("map^"))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase("^2") )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase("max") || opcode.equalsIgnoreCase("mapmax") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase("min") || opcode.equalsIgnoreCase("mapmin") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		
		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}
	
	public static String deriveAggregateOperatorOpcode(String opcode) {
		switch( opcode ) {
			case "uak+":
			case"uark+":
			case "uack+":    return "ak+";
			case "ua+":
			case "uar+":
			case "uac+":     return "a+";
			case "uatrace":
			case "uaktrace": return "aktrace";
			case "uasqk+":
			case "uarsqk+":
			case "uacsqk+":  return "asqk+";
			case "uamean":
			case "uarmean":
			case "uacmean":  return "amean";
			case "uavar":
			case "uarvar":
			case "uacvar":   return "avar";
			case "ua*":
			case "uar*":
			case "uac*":     return "a*";
			case "uamax":
			case "uarmax":
			case "uacmax":   return "amax";
			case "uamin":
			case "uarmin":
			case "uacmin":   return "amin";
			case "uarimax":  return "arimax";
			case "uarimin":  return "arimin";
		}
		return null;
	}
	
	public static AggOp getAggOp(String opcode) {
		switch( opcode ) {
			case "uak+":
			case"uark+":
			case "uack+":
			case "ua+":
			case "uar+":
			case "uac+":
			case "uatrace":
			case "uaktrace": return AggOp.SUM;
			case "uasqk+":
			case "uarsqk+":
			case "uacsqk+":  return AggOp.SUM_SQ;
			case "uamean":
			case "uarmean":
			case "uacmean":  return AggOp.MEAN;
			case "uavar":
			case "uarvar":
			case "uacvar":   return AggOp.VAR;
			case "ua*":
			case "uar*":
			case "uac*":     return AggOp.PROD;
			case "uamax":
			case "uarmax":
			case "uacmax":   return AggOp.MAX;
			case "uamin":
			case "uarmin":
			case "uacmin":   return AggOp.MIN;
			case "uarimax":  return AggOp.MAXINDEX;
			case "uarimin":  return AggOp.MININDEX;
		}
		return null;
	}
	
	public static Direction getAggDirection(String opcode) {
		switch( opcode ) {
			case "uak+":
			case "ua+":
			case "uatrace":
			case "uaktrace":
			case "uasqk+":
			case "uamean":
			case "uavar":
			case "ua*":
			case "uamax":
			case "uamin":    return Direction.RowCol;
			case"uark+":
			case "uar+":
			case "uarsqk+":
			case "uarmean":
			case "uar*":
			case "uarmax":
			case "uarmin":
			case "uarimax":
			case "uarimin":  return Direction.Row;
			case "uack+":
			case "uac+":
			case "uacsqk+":
			case "uacmean":
			case "uarvar":
			case "uacvar":
			case "uac*":
			case "uacmax":
			case "uacmin":   return Direction.Col;
		}
		return null;
	}

	public static CorrectionLocationType deriveAggregateOperatorCorrectionLocation(String opcode)
	{
		if ( opcode.equalsIgnoreCase("uak+") || opcode.equalsIgnoreCase("uark+") ||
				opcode.equalsIgnoreCase("uasqk+") || opcode.equalsIgnoreCase("uarsqk+") ||
				opcode.equalsIgnoreCase("uatrace") || opcode.equalsIgnoreCase("uaktrace") )
			return CorrectionLocationType.LASTCOLUMN;
		else if ( opcode.equalsIgnoreCase("uack+") || opcode.equalsIgnoreCase("uacsqk+") )
			return CorrectionLocationType.LASTROW;
		else if ( opcode.equalsIgnoreCase("uamean") || opcode.equalsIgnoreCase("uarmean") )
			return CorrectionLocationType.LASTTWOCOLUMNS;
		else if ( opcode.equalsIgnoreCase("uacmean") )
			return CorrectionLocationType.LASTTWOROWS;
		else if ( opcode.equalsIgnoreCase("uavar") || opcode.equalsIgnoreCase("uarvar") )
			return CorrectionLocationType.LASTFOURCOLUMNS;
		else if ( opcode.equalsIgnoreCase("uacvar") )
			return CorrectionLocationType.LASTFOURROWS;
		else if (opcode.equalsIgnoreCase("uarimax") || opcode.equalsIgnoreCase("uarimin") )
			return CorrectionLocationType.LASTCOLUMN;
		
		return CorrectionLocationType.NONE;
	}

	public static boolean isDistQuaternaryOpcode(String opcode) 
	{
		return WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode)     //mapwsloss
			|| WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode)    //redwsloss
			|| WeightedSigmoid.OPCODE.equalsIgnoreCase(opcode)   	   //mapwsigmoid
			|| WeightedSigmoidR.OPCODE.equalsIgnoreCase(opcode)        //redwsigmoid
			|| WeightedDivMM.OPCODE.equalsIgnoreCase(opcode)           //mapwdivmm
			|| WeightedDivMMR.OPCODE.equalsIgnoreCase(opcode)          //redwdivmm
			|| WeightedCrossEntropy.OPCODE.equalsIgnoreCase(opcode)    //mapwcemm
			|| WeightedCrossEntropyR.OPCODE.equalsIgnoreCase(opcode)   //redwcemm
			|| WeightedUnaryMM.OPCODE.equalsIgnoreCase(opcode)         //mapwumm
			|| WeightedUnaryMMR.OPCODE.equalsIgnoreCase(opcode);       //redwumm
	}
	
	public static AggregateBinaryOperator getMatMultOperator(int k) {
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		return new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, k);
	}
	
	public static Operator parseGroupedAggOperator(String fn, String other) {
		AggregateOperationTypes op = AggregateOperationTypes.INVALID;
		if ( fn.equalsIgnoreCase("centralmoment") )
			// in case of CM, we also need to pass "order"
			op = CMOperator.getAggOpType(fn, other);
		else 
			op = CMOperator.getAggOpType(fn, null);
	
		switch(op) {
		case SUM:
			return new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			
		case COUNT:
		case MEAN:
		case VARIANCE:
		case CM2:
		case CM3:
		case CM4:
			return new CMOperator(CM.getCMFnObject(op), op);
		case INVALID:
		default:
			throw new DMLRuntimeException("Invalid Aggregate Operation in GroupedAggregateInstruction: " + op);
		}
	}
	
	public static String replaceOperand(String instStr, int operand, String newValue) {
		//split instruction and check for correctness
		String[] parts = instStr.split(Lop.OPERAND_DELIMITOR);
		if( operand >= parts.length )
			throw new DMLRuntimeException("Operand position "
				+ operand + " exceeds the length of the instruction.");
		
		//replace and reconstruct string
		parts[operand] = newValue;
		StringBuilder sb = new StringBuilder(instStr.length());
		sb.append(parts[0]);
		for( int i=1; i<parts.length; i++ ) {
			sb.append(Lop.OPERAND_DELIMITOR);
			sb.append(parts[i]);
		}
		return sb.toString();
	}
	
	public static String concatOperands(String... inputs) {
		StringBuilder sb = new StringBuilder(64);
		for( int i=0; i<inputs.length-1; i++ ) {
			sb.append(inputs[i]);
			sb.append(Lop.OPERAND_DELIMITOR);
		}
		sb.append(inputs[inputs.length-1]);
		return sb.toString();
	}
}
