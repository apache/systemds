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

import java.util.Arrays;
import java.util.StringTokenizer;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.InstructionType;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.BitwOr;
import org.apache.sysds.runtime.functionobjects.BitwShiftL;
import org.apache.sysds.runtime.functionobjects.BitwShiftR;
import org.apache.sysds.runtime.functionobjects.BitwXor;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.IfElse;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Minus1Multiply;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.MinusNz;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Multiply2;
import org.apache.sysds.runtime.functionobjects.Not;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceDiag;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.Xor;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction.GPUINSTRUCTION_TYPE;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.MultiThreadedOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnarySketchOperator;


public class InstructionUtils {
	protected static final Log LOG = LogFactory.getLog(InstructionUtils.class.getName());

	//thread-local string builders for instruction concatenation (avoid allocation)
	private static ThreadLocal<StringBuilder> _strBuilders = new ThreadLocal<>() {
		@Override
		protected StringBuilder initialValue() { 
			return new StringBuilder(64);
		}
	};
	
	public static StringBuilder getStringBuilder() {
		StringBuilder sb = _strBuilders.get();
		sb.setLength(0); //reuse allocated space
		return sb;
	}
	
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

	public static int checkNumFields( String[] parts, int... expected ) {
		int numParts = parts.length;
		int numFields = numParts - 1; //account for opcode
		return checkMatchingNumField(numFields, expected);
	}

	private static int checkMatchingNumField(int numFields, int... expected){
		if (Arrays.stream(expected).noneMatch((i) -> numFields == i)) {
			StringBuilder sb = new StringBuilder();
			sb.append("checkNumFields() -- expected number (");
			for (int i = 0; i < expected.length; i++) {
				sb.append(expected[i]);
				if (i != expected.length - 1)
					sb.append(", ");
			}
			sb.append(") != is not equal to actual number (").append(numFields).append(").");
			throw new DMLRuntimeException(sb.toString());
		}
		return numFields;
	}

	public static int checkNumFields( String str, int... expected ) {
		int numParts = str.split(Instruction.OPERAND_DELIM).length;
		int numFields = numParts - 2; // -2 accounts for execType and opcode
		return checkMatchingNumField(numFields, expected);
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
	 * execution type (CP or SPARK) and returns the remaining parts, 
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

	public static String stripThreadCount(String str) {
		String[] parts = str.split(Instruction.OPERAND_DELIM, -1);
		String[] ret = new String[parts.length-1];
		for (int i=0; i<parts.length-1; i++) //strip-off the thread count
			ret[i] = parts[i];
		return concatOperands(ret);
	}

	public static ExecType getExecType( String str ) {
		try{
			int ix = str.indexOf(Instruction.OPERAND_DELIM);
			return ExecType.valueOf(str.substring(0, ix));
		}
		catch(Exception e){
			throw new DMLRuntimeException("Unable to extract Execution type from " + str, e);
		}
	}

	public static String getOpCode( String str ) {
		int ix1 = str.indexOf(Instruction.OPERAND_DELIM);
		int ix2 = str.indexOf(Instruction.OPERAND_DELIM, ix1+1);
		return str.substring(ix1+1, ix2);
	}

	public static InstructionType getSPType(String str) {
		String opcode = getOpCode(str);
		return Opcodes.getTypeByOpcode(opcode, Types.ExecType.SPARK);
	}

	public static InstructionType getCPType(String str) {
		String opcode = getOpCode(str);
		return Opcodes.getTypeByOpcode(opcode, Types.ExecType.CP);
	}
	
	public static InstructionType getSPTypeByOpcode(String opcode) {
		String op = getOpCode(opcode);
		return Opcodes.getTypeByOpcode(op, Types.ExecType.SPARK);
	}
	
	public static InstructionType getCPTypeByOpcode(String opcode ) {
		String op = getOpCode(opcode);
		return Opcodes.getTypeByOpcode(op, Types.ExecType.CP);
	}

	public static GPUINSTRUCTION_TYPE getGPUType( String str ) {
		return GPUInstructionParser.String2GPUInstructionType.get(getOpCode(str));
	}
	
	public static InstructionType getFEDType(String str) {
		String op = getOpCode(str);
		return Opcodes.getTypeByOpcode(op, Types.ExecType.FED);
	}

	public static InstructionType getOOCType(String str) {
		String op = getOpCode(str);
		return Opcodes.getTypeByOpcode(op, Types.ExecType.OOC);
	}

	public static boolean isBuiltinFunction( String opcode ) {
		Builtin.BuiltinCode bfc = Builtin.String2BuiltinCode.get(opcode);
		return (bfc != null);
	}
	
	public static boolean isUnaryMetadata(String opcode) {
		return opcode != null 
			&& (opcode.equals("nrow") || opcode.equals("ncol"));
	}
	
	public static AggregateUnaryOperator parseBasicAggregateUnaryOperator(String opcode) {
		return parseBasicAggregateUnaryOperator(opcode, 1);
	}

	/**
	 * Parse the given opcode into an aggregate unary operator.
	 *
	 * @param opcode opcode
	 * @param numThreads number of threads
	 * @return Parsed aggregate unary operator object. Caller must handle possible null return value.
	 */
	public static AggregateUnaryOperator parseBasicAggregateUnaryOperator(String opcode, int numThreads)
	{
		AggregateUnaryOperator aggun = null;
		
		if ( opcode.equalsIgnoreCase(Opcodes.UAKP.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UARKP.toString()) ) { // RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UACKP.toString()) ) { // ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTROW);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UASQKP.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UARSQKP.toString()) ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACSQKP.toString()) ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), CorrectionLocationType.LASTROW);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UAMEAN.toString()) ) {
			// Mean
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), CorrectionLocationType.LASTTWOCOLUMNS);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UARMEAN.toString()) ) {
			// RowMeans
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), CorrectionLocationType.LASTTWOCOLUMNS);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UACMEAN.toString()) ) {
			// ColMeans
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), CorrectionLocationType.LASTTWOROWS);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UAVAR.toString()) ) {
			// Variance
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			CorrectionLocationType cloc = CorrectionLocationType.LASTFOURCOLUMNS;
			AggregateOperator agg = new AggregateOperator(0, varFn, cloc);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UARVAR.toString()) ) {
			// RowVariances
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			CorrectionLocationType cloc = CorrectionLocationType.LASTFOURCOLUMNS;
			AggregateOperator agg = new AggregateOperator(0, varFn, cloc);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACVAR.toString()) ) {
			// ColVariances
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			CorrectionLocationType cloc = CorrectionLocationType.LASTFOURROWS;
			AggregateOperator agg = new AggregateOperator(0, varFn, cloc);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UAP.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UARP.toString()) ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UACP.toString()) ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UAM.toString()) ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UARM.toString()) ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UACM.toString()) ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UAMAX.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UAMIN.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UATRACE.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UAKTRACE.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject(), numThreads);
		} 		
		else if ( opcode.equalsIgnoreCase(Opcodes.UARMAX.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if (opcode.equalsIgnoreCase(Opcodes.UARIMAX.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("maxindex"), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UARMIN.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if (opcode.equalsIgnoreCase(Opcodes.UARIMIN.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("minindex"), CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACMAX.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.UACMIN.toString()) ) {
			AggregateOperator agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACD.toString()) ) {
			aggun = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT,
					Direction.RowCol, ReduceAll.getReduceAllFnObject());
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACDR.toString()) ) {
			aggun = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT,
					Direction.Row, ReduceCol.getReduceColFnObject());
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACDC.toString()) ) {
			aggun = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT,
					Direction.Col, ReduceRow.getReduceRowFnObject());
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACDAP.toString()) ) {
			aggun = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT_APPROX,
					Direction.RowCol, ReduceAll.getReduceAllFnObject());
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACDAPR.toString()) ) {
			aggun = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT_APPROX,
					Direction.Row, ReduceCol.getReduceColFnObject());
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UACDAPC.toString()) ) {
			aggun = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT_APPROX,
					Direction.Col, ReduceRow.getReduceRowFnObject());
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UNIQUE.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("unique"));
			aggun = new UnarySketchOperator(agg, ReduceAll.getReduceAllFnObject(), Direction.RowCol, numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UNIQUER.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("unique"));
			aggun = new UnarySketchOperator(agg, ReduceCol.getReduceColFnObject(), Direction.Row, numThreads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.UNIQUEC.toString()) ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("unique"));
			aggun = new UnarySketchOperator(agg, ReduceRow.getReduceRowFnObject(), Direction.Col, numThreads);
		}

		return aggun;
	}

	public static AggregateUnaryOperator parseAggregateUnaryRowIndexOperator(String opcode, int numOutputs, int numThreads) {
		AggregateUnaryOperator aggun = null;
		AggregateOperator agg = null;
		if (opcode.equalsIgnoreCase(Opcodes.UARIMAX.toString()) )
			agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("maxindex"),
				numOutputs == 1 ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.NONE);

		else if (opcode.equalsIgnoreCase(Opcodes.UARIMIN.toString()) )
			agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("minindex"),
				numOutputs == 1 ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.NONE);

		aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numThreads);
		return aggun;
	}

	public static AggregateTernaryOperator parseAggregateTernaryOperator(String opcode) {
		return parseAggregateTernaryOperator(opcode, 1);
	}
	
	public static AggregateTernaryOperator parseAggregateTernaryOperator(String opcode, int numThreads) {
		CorrectionLocationType corr = opcode.equalsIgnoreCase(Opcodes.TAKPM.toString()) ?
			CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.LASTROW;
		AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corr);
		IndexFunction ixfun = opcode.equalsIgnoreCase(Opcodes.TAKPM.toString()) ?
			ReduceAll.getReduceAllFnObject() : ReduceRow.getReduceRowFnObject();
		
		return new AggregateTernaryOperator(Multiply.getMultiplyFnObject(), agg, ixfun, numThreads);
	}
	
	public static AggregateOperator parseAggregateOperator(String opcode, String corrLoc)
	{
		AggregateOperator agg = null;
	
		if ( opcode.equalsIgnoreCase("ak+") || opcode.equalsIgnoreCase("aktrace") ) {
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), lcorrLoc);
		}
		else if ( opcode.equalsIgnoreCase("asqk+") ) {
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject(), lcorrLoc);
		}
		else if ( opcode.equalsIgnoreCase("a+") ) {
			agg = new AggregateOperator(0, Plus.getPlusFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("a*") ) {
			agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
		}
		else if (opcode.equalsIgnoreCase("arimax")){
			agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("maxindex"), CorrectionLocationType.LASTCOLUMN);
		}
		else if ( opcode.equalsIgnoreCase("amax") ) {
			agg = new AggregateOperator(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject("max"));
		}
		else if ( opcode.equalsIgnoreCase("amin") ) {
			agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("min"));
		}
		else if (opcode.equalsIgnoreCase("arimin")){
			agg = new AggregateOperator(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject("minindex"), CorrectionLocationType.LASTCOLUMN);
		}
		else if ( opcode.equalsIgnoreCase("amean") ) {
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTTWOCOLUMNS : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), lcorrLoc);
		}
		else if ( opcode.equalsIgnoreCase("avar") ) {
			CorrectionLocationType lcorrLoc = (corrLoc==null) ?
				CorrectionLocationType.LASTFOURCOLUMNS :
				CorrectionLocationType.valueOf(corrLoc);
			CM varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);
			agg = new AggregateOperator(0, varFn, lcorrLoc);
		}

		return agg;
	}

	public static AggregateUnaryOperator parseBasicCumulativeAggregateUnaryOperator(UnaryOperator uop) {
		Builtin f = (Builtin)uop.fn;
		if( f.getBuiltinCode()==BuiltinCode.CUMSUM ) 
			return parseBasicAggregateUnaryOperator(Opcodes.UACKP.toString()) ;
		else if( f.getBuiltinCode()==BuiltinCode.ROWCUMSUM )
			return parseBasicAggregateUnaryOperator(Opcodes.UARCKP.toString()) ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMPROD ) 
			return parseBasicAggregateUnaryOperator(Opcodes.UACM.toString()) ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMMIN ) 
			return parseBasicAggregateUnaryOperator(Opcodes.UACMIN.toString()) ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMMAX ) 
			return parseBasicAggregateUnaryOperator(Opcodes.UACMAX.toString() ) ;
		else if( f.getBuiltinCode()==BuiltinCode.CUMSUMPROD ) 
			return parseBasicAggregateUnaryOperator("uack+*" ) ;
		throw new RuntimeException("Unsupported cumulative aggregate unary operator: "+f.getBuiltinCode());
	}

	public static AggregateUnaryOperator parseCumulativeAggregateUnaryOperator(String opcode) {
		AggregateOperator agg = null;
		if( Opcodes.UCUMACKP.toString().equals(opcode) )
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTROW);
		else if ( Opcodes.UCUMACM.toString().equals(opcode) )
			agg = new AggregateOperator(1, Multiply.getMultiplyFnObject(), CorrectionLocationType.NONE);
		else if ( Opcodes.UCUMACPM.toString().equals(opcode) )
			agg = new AggregateOperator(0, PlusMultiply.getFnObject(), CorrectionLocationType.NONE);
		else if ( Opcodes.UCUMACMIN.toString().equals(opcode) )
			agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"), CorrectionLocationType.NONE);
		else if ( Opcodes.UCUMACMAX.toString().equals(opcode) )
			agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"), CorrectionLocationType.NONE);
		return new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
	}
	
	public static UnaryOperator parseUnaryOperator(String opcode) {
		return opcode.equals(Opcodes.NOT.toString()) ?
			new UnaryOperator(Not.getNotFnObject()) :
			new UnaryOperator(Builtin.getBuiltinFnObject(opcode));
	}

	public static UnaryOperator parseUnaryOperator(String opcode, int k) {
		return opcode.equals(Opcodes.NOT.toString()) ?
			new UnaryOperator(Not.getNotFnObject(), k) :
			new UnaryOperator(Builtin.getBuiltinFnObject(opcode), k);
	}

	public static MultiThreadedOperator parseBinaryOrBuiltinOperator(String opcode, CPOperand in1, CPOperand in2) {
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
		boolean matrixScalar = (in1.getDataType() != in2.getDataType() && (in1.getDataType() != Types.DataType.FRAME && in2.getDataType() != Types.DataType.FRAME));
		return Builtin.isBuiltinFnObject(opcode) ?
			(matrixScalar ? new RightScalarOperator( Builtin.getBuiltinFnObject(opcode), 0) :
				new BinaryOperator( Builtin.getBuiltinFnObject(opcode))) :
			(matrixScalar ? parseScalarBinaryOperator(opcode, in1.getDataType().isScalar()) :
				parseExtendedBinaryOperator(opcode));

	}

	public static BinaryOperator parseBinaryOperator(String opcode, int k){
		BinaryOperator bop = parseBinaryOperator(opcode);
		bop.setNumThreads(k);
		return  bop;
	}

	public static BinaryOperator parseBinaryOperator(String opcode)
	{
		if(opcode.equalsIgnoreCase(Opcodes.EQUAL.toString()))
			return new BinaryOperator(Equals.getEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.NOTEQUAL.toString()))
			return new BinaryOperator(NotEquals.getNotEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.LESS.toString()))
			return new BinaryOperator(LessThan.getLessThanFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.GREATER.toString()))
			return new BinaryOperator(GreaterThan.getGreaterThanFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.LESSEQUAL.toString()))
			return new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.GREATEREQUAL.toString()))
			return new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.AND.toString()))
			return new BinaryOperator(And.getAndFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.OR.toString()))
			return new BinaryOperator(Or.getOrFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.XOR.toString()))
			return new BinaryOperator(Xor.getXorFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWAND.toString()))
			return new BinaryOperator(BitwAnd.getBitwAndFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWOR.toString()))
			return new BinaryOperator(BitwOr.getBitwOrFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWXOR.toString()))
			return new BinaryOperator(BitwXor.getBitwXorFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWSHIFTL.toString()))
			return new BinaryOperator(BitwShiftL.getBitwShiftLFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWSHIFTR.toString()))
			return new BinaryOperator(BitwShiftR.getBitwShiftRFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.PLUS.toString()))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MINUS.toString()))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MULT.toString()))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MINUS1_MULT.toString()))
			return new BinaryOperator(Minus1Multiply.getMinus1MultiplyFnObject());
		else if ( opcode.equalsIgnoreCase(Opcodes.MULT2.toString()) )
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.DIV.toString()))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MODULUS.toString()))
			return new BinaryOperator(Modulus.getFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.INTDIV.toString()))
			return new BinaryOperator(IntegerDivide.getFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.POW.toString()))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase(Opcodes.POW2.toString()) )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase(Opcodes.MAX.toString()) )
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase(Opcodes.MIN.toString()) )
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		else if( opcode.equalsIgnoreCase(Opcodes.DROPINVALIDTYPE.toString()))
			return new BinaryOperator(Builtin.getBuiltinFnObject("dropInvalidType"));
		else if( opcode.equalsIgnoreCase(Opcodes.DROPINVALIDLENGTH.toString()))
			return new BinaryOperator(Builtin.getBuiltinFnObject("dropInvalidLength"));
		else if( opcode.equalsIgnoreCase(Opcodes.VALUESWAP.toString()))
			return new BinaryOperator(Builtin.getBuiltinFnObject("valueSwap"));
		else if( opcode.equalsIgnoreCase(Opcodes.FREPLICATE.toString()))
			return new BinaryOperator(Builtin.getBuiltinFnObject("freplicate"));

		throw new RuntimeException("Unknown binary opcode " + opcode);
	}
	
	public static TernaryOperator parseTernaryOperator(String opcode) {
		return parseTernaryOperator(opcode, 1);
	}
	
	public static TernaryOperator parseTernaryOperator(String opcode, int numThreads) {
		return new TernaryOperator(opcode.equals(Opcodes.PM.toString()) ? PlusMultiply.getFnObject() :
			opcode.equals(Opcodes.MINUSMULT.toString()) ? MinusMultiply.getFnObject() : IfElse.getFnObject(), numThreads);
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
		// TODO add Multithreaded threads to Scalar operations.

		//commutative operators
		if ( opcode.equalsIgnoreCase(Opcodes.PLUS.toString()) ){
			return new RightScalarOperator(Plus.getPlusFnObject(), constant); 
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.MULT.toString()) ) {
			return new RightScalarOperator(Multiply.getMultiplyFnObject(), constant);
		} 
		//non-commutative operators
		else if ( opcode.equalsIgnoreCase(Opcodes.MINUS.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Minus.getMinusFnObject(), constant);
			else return new RightScalarOperator(Minus.getMinusFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.MINUS_NZ.toString()) ) {
			//no support for left scalar yet
			return new RightScalarOperator(MinusNz.getMinusNzFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.DIV.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Divide.getDivideFnObject(), constant);
			else return new RightScalarOperator(Divide.getDivideFnObject(), constant);
		}  
		else if ( opcode.equalsIgnoreCase(Opcodes.MODULUS.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Modulus.getFnObject(), constant);
			else return new RightScalarOperator(Modulus.getFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.INTDIV.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(IntegerDivide.getFnObject(), constant);
			else return new RightScalarOperator(IntegerDivide.getFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.POW.toString()) ){
			if(arg1IsScalar)
				return new LeftScalarOperator(Power.getPowerFnObject(), constant);
			else return new RightScalarOperator(Power.getPowerFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.MAX.toString()) ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("max"), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.MIN.toString()) ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("min"), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.LOG.toString()) || opcode.equalsIgnoreCase(Opcodes.LOGNZ.toString()) ){
			if( arg1IsScalar )
				return new LeftScalarOperator(Builtin.getBuiltinFnObject(opcode), constant);
			return new RightScalarOperator(Builtin.getBuiltinFnObject(opcode), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.GREATER.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(GreaterThan.getGreaterThanFnObject(), constant);
			return new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.GREATEREQUAL.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), constant);
			return new RightScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.LESS.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(LessThan.getLessThanFnObject(), constant);
			return new RightScalarOperator(LessThan.getLessThanFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.LESSEQUAL.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), constant);
			return new RightScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.EQUAL.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Equals.getEqualsFnObject(), constant);
			return new RightScalarOperator(Equals.getEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.NOTEQUAL.toString()) ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(NotEquals.getNotEqualsFnObject(), constant);
			return new RightScalarOperator(NotEquals.getNotEqualsFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.AND.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(And.getAndFnObject(), constant) :
				new RightScalarOperator(And.getAndFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.OR.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(Or.getOrFnObject(), constant) :
				new RightScalarOperator(Or.getOrFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.XOR.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(Xor.getXorFnObject(), constant) :
				new RightScalarOperator(Xor.getXorFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.BITWAND.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwAnd.getBitwAndFnObject(), constant) :
				new RightScalarOperator(BitwAnd.getBitwAndFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.BITWOR.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwOr.getBitwOrFnObject(), constant) :
				new RightScalarOperator(BitwOr.getBitwOrFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.BITWXOR.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwXor.getBitwXorFnObject(), constant) :
				new RightScalarOperator(BitwXor.getBitwXorFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.BITWSHIFTL.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwShiftL.getBitwShiftLFnObject(), constant) :
				new RightScalarOperator(BitwShiftL.getBitwShiftLFnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.BITWSHIFTR.toString()) ) {
			return arg1IsScalar ?
				new LeftScalarOperator(BitwShiftR.getBitwShiftRFnObject(), constant) :
				new RightScalarOperator(BitwShiftR.getBitwShiftRFnObject(), constant);
		}
		//operations that only exist for performance purposes (all unary or commutative operators)
		else if ( opcode.equalsIgnoreCase(Opcodes.MULT2.toString()) ) {
			return new RightScalarOperator(Multiply2.getMultiply2FnObject(), constant);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.POW2.toString()) ){
			return new RightScalarOperator(Power2.getPower2FnObject(), constant);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.MINUS1_MULT.toString()) ) {
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
		if(opcode.equalsIgnoreCase(Opcodes.EQUAL.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPEQ.toString()))
			return new BinaryOperator(Equals.getEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.NOTEQUAL.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPNEQ.toString()))
			return new BinaryOperator(NotEquals.getNotEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.LESS.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPLT.toString()))
			return new BinaryOperator(LessThan.getLessThanFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.GREATER.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPGT.toString()))
			return new BinaryOperator(GreaterThan.getGreaterThanFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.LESSEQUAL.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPLE.toString()))
			return new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.GREATEREQUAL.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPGE.toString()))
			return new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.AND.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPAND.toString()))
			return new BinaryOperator(And.getAndFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.OR.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPOR.toString()))
			return new BinaryOperator(Or.getOrFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.XOR.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPXOR.toString()))
			return new BinaryOperator(Xor.getXorFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWAND.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPBITWAND.toString()))
			return new BinaryOperator(BitwAnd.getBitwAndFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWOR.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPBITWOR.toString()))
			return new BinaryOperator(BitwOr.getBitwOrFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWXOR.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPBITWXOR.toString()))
			return new BinaryOperator(BitwXor.getBitwXorFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWSHIFTL.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPBITWSHIFTL.toString()))
			return new BinaryOperator(BitwShiftL.getBitwShiftLFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.BITWSHIFTR.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPBITWSHIFTR.toString()))
			return new BinaryOperator(BitwShiftR.getBitwShiftRFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.PLUS.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPPLUS.toString()))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MINUS.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPMINUS.toString()))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MULT.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPMULT.toString()))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MINUS1_MULT.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPMINUS1_MULT.toString()))
			return new BinaryOperator(Minus1Multiply.getMinus1MultiplyFnObject());
		else if ( opcode.equalsIgnoreCase(Opcodes.MULT2.toString()) )
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.DIV.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPDIV.toString()))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.MODULUS.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPMOD.toString()))
			return new BinaryOperator(Modulus.getFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.INTDIV.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPINTDIV.toString()))
			return new BinaryOperator(IntegerDivide.getFnObject());
		else if(opcode.equalsIgnoreCase(Opcodes.POW.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPPOW.toString()))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase(Opcodes.POW2.toString()) )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase(Opcodes.MAX.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPMAX.toString()) )
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase(Opcodes.MIN.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPMIN.toString()) )
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		else if ( opcode.equalsIgnoreCase(Opcodes.DROPINVALIDLENGTH.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPDROPINVALIDLENGTH.toString()) )
			return new BinaryOperator(Builtin.getBuiltinFnObject("dropInvalidLength"));
		else if ( opcode.equalsIgnoreCase(Opcodes.VALUESWAP.toString()) || opcode.equalsIgnoreCase("mapValueSwap") )
			return new BinaryOperator(Builtin.getBuiltinFnObject("valueSwap"));

		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}

	public static ScalarOperator parseScalarBinaryOperator(String opcode, boolean arg1IsScalar, double constant, int k){
		ScalarOperator sop = parseScalarBinaryOperator(opcode, arg1IsScalar, constant);
		sop.setNumThreads(k);
		return sop;
	}
	
	public static String deriveAggregateOperatorOpcode(String opcode) {
		switch( opcode ) {
			case "uak+":
			case"uark+":
			case "uack+":    return "ak+";
			case "uacd":
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
		if ( opcode.equalsIgnoreCase(Opcodes.UAKP.toString()) || opcode.equalsIgnoreCase(Opcodes.UARKP.toString()) ||
				opcode.equalsIgnoreCase(Opcodes.UASQKP.toString()) || opcode.equalsIgnoreCase(Opcodes.UARSQKP.toString()) ||
				opcode.equalsIgnoreCase(Opcodes.UATRACE.toString()) || opcode.equalsIgnoreCase(Opcodes.UAKTRACE.toString()) )
			return CorrectionLocationType.LASTCOLUMN;
		else if ( opcode.equalsIgnoreCase(Opcodes.UACKP.toString()) || opcode.equalsIgnoreCase(Opcodes.UACSQKP.toString()) )
			return CorrectionLocationType.LASTROW;
		else if ( opcode.equalsIgnoreCase(Opcodes.UAMEAN.toString()) || opcode.equalsIgnoreCase(Opcodes.UARMEAN.toString()) )
			return CorrectionLocationType.LASTTWOCOLUMNS;
		else if ( opcode.equalsIgnoreCase(Opcodes.UACMEAN.toString()) )
			return CorrectionLocationType.LASTTWOROWS;
		else if ( opcode.equalsIgnoreCase(Opcodes.UAVAR.toString()) || opcode.equalsIgnoreCase(Opcodes.UARVAR.toString()) )
			return CorrectionLocationType.LASTFOURCOLUMNS;
		else if ( opcode.equalsIgnoreCase(Opcodes.UACVAR.toString()) )
			return CorrectionLocationType.LASTFOURROWS;
		else if (opcode.equalsIgnoreCase(Opcodes.UARIMAX.toString()) || opcode.equalsIgnoreCase(Opcodes.UARIMIN.toString()) )
			return CorrectionLocationType.LASTCOLUMN;
		
		return CorrectionLocationType.NONE;
	}

	public static boolean isDistQuaternaryOpcode(String opcode) 
	{
		return Opcodes.WEIGHTEDSQUAREDLOSS.toString().equalsIgnoreCase(opcode)     //mapwsloss
			|| Opcodes.WEIGHTEDSQUAREDLOSSR.toString().equalsIgnoreCase(opcode)    //redwsloss
			|| Opcodes.WEIGHTEDSIGMOID.toString().equalsIgnoreCase(opcode)   	   //mapwsigmoid
			|| Opcodes.WEIGHTEDSIGMOIDR.toString().equalsIgnoreCase(opcode)        //redwsigmoid
			|| Opcodes.WEIGHTEDDIVMM.toString().equalsIgnoreCase(opcode)           //mapwdivmm
			|| Opcodes.WEIGHTEDDIVMMR.toString().equalsIgnoreCase(opcode)          //redwdivmm
			|| Opcodes.WEIGHTEDCROSSENTROPY.toString().equalsIgnoreCase(opcode)    //mapwcemm
			|| Opcodes.WEIGHTEDCROSSENTROPYR.toString().equalsIgnoreCase(opcode)   //redwcemm
			|| Opcodes.WEIGHTEDUNARYMM.toString().equalsIgnoreCase(opcode)         //mapwumm
			|| Opcodes.WEIGHTEDUNARYMMR.toString().equalsIgnoreCase(opcode);       //redwumm
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
				return new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.LASTCOLUMN);
				
			case COUNT:
			case MEAN:
			case VARIANCE:
			case CM2:
			case CM3:
			case CM4:
			
			//TODO use appropriate function objects for min/max (see sum)
			case MIN:
			case MAX:
				return new CMOperator(CM.getCMFnObject(op), op);
			case INVALID:
			default:
				throw new DMLRuntimeException("Invalid Aggregate Operation in GroupedAggregateInstruction: " + op);
		}
	}
	
	public static String createLiteralOperand(String val, ValueType vt) {
		return InstructionUtils.concatOperandParts(val, DataType.SCALAR.name(), vt.name(), "true");
	}

	public static String createOperand(CPOperand operand) {
		return InstructionUtils.concatOperandParts(operand.getName(), operand.getDataType().name(), operand.getValueType().name());
	}

	public static String replaceOperand(String instStr, int operand, String newValue) {
		//split instruction and check for correctness
		String[] parts = instStr.split(Lop.OPERAND_DELIMITOR);
		if( operand >= parts.length )
			throw new DMLRuntimeException("Operand position "
				+ operand + " exceeds the length of the instruction.");
		//replace and reconstruct string
		parts[operand] = newValue;
		return concatOperands(parts);
	}

	public static String removeOperand(String instStr, int operand) {
		//split instruction and check for correctness
		String[] parts = instStr.split(Lop.OPERAND_DELIMITOR);
		if( operand >= parts.length )
			throw new DMLRuntimeException("Operand position "
				+ operand + " exceeds the length of the instruction.");
		//remove and reconstruct string
		return concatOperands(ArrayUtils.remove(parts, operand));
	}

	public static String replaceOperandName(String instStr) {
		String[] parts = instStr.split(Lop.OPERAND_DELIMITOR);
		String oldName = parts[parts.length-1];
		String[] Nameparts = oldName.split(Instruction.VALUETYPE_PREFIX);
		Nameparts[0] = "xxx";
		String newName = concatOperandParts(Nameparts);
		parts[parts.length-1] = newName;
		return concatOperands(parts);
	}

	/**
	 * Concat the inputs as operands to generate the instruction string.
	 * The inputs are separated by the operand delimiter and appended
	 * using a ThreadLocal StringBuilder.
	 * @param inputs operand inputs given as strings
	 * @return the instruction string with the given inputs concatenated
	 */
	public static String concatOperands(String... inputs) {
		StringBuilder sb = _strBuilders.get();
		sb.setLength(0); //reuse allocated space
		return concatOperands(sb, inputs);
	}
	
	public static String concatOperands(StringBuilder sb, String... inputs) {
		return concatBaseOperandsWithDelim(sb, Lop.OPERAND_DELIMITOR, inputs);
	}

	/**
	 * Concat the input parts with the value type delimiter.
	 * @param inputs input operand parts as strings
	 * @return concatenated input parts
	 */
	public static String concatOperandParts(String... inputs) {
		StringBuilder sb = _strBuilders.get();
		sb.setLength(0); //reuse allocated space
		return concatBaseOperandsWithDelim(sb, Instruction.VALUETYPE_PREFIX, inputs);
	}

	private static String concatBaseOperandsWithDelim(StringBuilder sb, String delim, String... inputs){
		for( int i=0; i<inputs.length-1; i++ ) {
			sb.append(inputs[i]);
			sb.append(delim);
		}
		sb.append(inputs[inputs.length-1]);
		return sb.toString();
	}
	
	public static String concatStrings(String... inputs) {
		StringBuilder sb = _strBuilders.get();
		sb.setLength(0); //reuse allocated space
		for( int i=0; i<inputs.length; i++ )
			sb.append(inputs[i]);
		return sb.toString();
	}

	public static String constructTernaryString(String instString, CPOperand op1, CPOperand op2, CPOperand op3, CPOperand out) {
		return concatOperands(constructBinaryInstString(instString, "ifelse", op1, op2, op3), createOperand(out));
	}

	public static String constructBinaryInstString(String instString, String opcode, CPOperand op1, CPOperand op2, CPOperand out) {
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		return InstructionUtils.concatOperands(parts[0], opcode, createOperand(op1), createOperand(op2), createOperand(out));
	}

	public static String constructUnaryInstString(String instString, String opcode, CPOperand op1, CPOperand out) {
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		return InstructionUtils.concatOperands(parts[0], opcode, createOperand(op1), createOperand(out));
	}

	/**
	 * Prepare instruction string for sending in a FederatedRequest as a CP instruction.
	 * This involves replacing the coordinator operand names with the worker operand names,
	 * changing the execution type, and removing the federated output flag if necessary.
	 * @param inst instruction string to prepare for federated request
	 * @param varOldOut current output operand (to be replaced)
	 * @param id new output operand (always a number)
	 * @param varOldIn current input operand (to be replaced)
	 * @param varNewIn new input operand names (always numbers)
	 * @param rmFederatedOutput remove federated output flag
	 * @return instruction string prepared for federated request
	 */
	public static String instructionStringFEDPrepare(String inst, CPOperand varOldOut, long id, CPOperand[] varOldIn, long[] varNewIn, boolean rmFederatedOutput){
		boolean isFedInstr = inst.startsWith(ExecType.FED.name() + Lop.OPERAND_DELIMITOR);
		String linst = replaceExecTypeWithCP(inst);
		linst = replaceOutputOperand(linst, varOldOut, id);
		linst = replaceInputOperand(linst, varOldIn, varNewIn);
		if(rmFederatedOutput && isFedInstr)
			linst = removeFEDOutputFlag(linst);
		return linst;
	}

	private static String replaceExecTypeWithCP(String inst){
		// Replace FED with CP instruction when it is send over to federated site.
		return inst.replace(Types.ExecType.FED.name(), Types.ExecType.CP.name());
	}

	private static String replaceOutputOperand(String linst, CPOperand varOldOut, long id){
		return replaceOperand(linst, varOldOut, Long.toString(id));
	}

	private static String replaceInputOperand(String linst, CPOperand[] varOldIn, long[] varNewIn){
		for(int i=0; i<varOldIn.length; i++)
			if( varOldIn[i] != null ) {
				linst = replaceOperand(linst, varOldIn[i], Long.toString(varNewIn[i]));
				linst = linst.replace("="+varOldIn[i].getName(), "="+varNewIn[i]); //parameterized
			}
		return linst;
	}

	/**
	 * Removes federated output flag from the end of the instruction string if the flag is present.
	 * @param linst instruction string
	 * @return instruction string with no federated output flag
	 */
	public static String removeFEDOutputFlag(String linst){
		int lastOperandStartIndex = linst.lastIndexOf(Lop.OPERAND_DELIMITOR);
		String lastOperand = linst.substring(lastOperandStartIndex);
		if ( containsFEDOutputFlag(lastOperand) )
			return linst.substring(0, lastOperandStartIndex);
		else return linst;
	}

	/**
	 * Checks whether the given operand string contains a federated output flag
	 * @param operandString which is checked for federated output flag
	 * @return true if the given operand string contains a federated output flag
	 */
	private static boolean containsFEDOutputFlag(String operandString){
		for (FederatedOutput fedOutput : FederatedOutput.values()){
			if ( operandString.contains(fedOutput.name()) )
				return true;
		}
		return false;
	}

	private static String replaceOperand(String linst, CPOperand oldOperand, String newOperandName){
		return linst.replace(
			Lop.OPERAND_DELIMITOR+oldOperand.getName()+Lop.DATATYPE_PREFIX,
			Lop.OPERAND_DELIMITOR+newOperandName+Lop.DATATYPE_PREFIX);
	}

	protected static boolean isInteger(String s){
		if(s.isEmpty()) return false;
		for(int i = 0; i < s.length(); i++) {
			 if(i == 0 && s.charAt(i) == '-') {
				  if(s.length() == 1) return false;
				  else continue;
			 }
			 if(Character.digit(s.charAt(i),10) < 0) return false;
		}
		return true;
	}
}
