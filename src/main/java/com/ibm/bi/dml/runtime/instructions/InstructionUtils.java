/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import java.util.StringTokenizer;

import com.ibm.bi.dml.lops.AppendM;
import com.ibm.bi.dml.lops.BinaryM;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.MapMultChain;
import com.ibm.bi.dml.lops.PMMJ;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.UAggOuterChain;
import com.ibm.bi.dml.lops.WeightedSigmoid;
import com.ibm.bi.dml.lops.WeightedSigmoidR;
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.lops.WeightedSquaredLossR;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.IntegerDivide;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.Mean;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.MinusNz;
import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Multiply2;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.functionobjects.Power2;
import com.ibm.bi.dml.runtime.functionobjects.ReduceAll;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.functionobjects.ReduceDiag;
import com.ibm.bi.dml.runtime.functionobjects.ReduceRow;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction.MRINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction.SPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.LeftScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.RightScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;


public class InstructionUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @param str
	 * @param expected
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static int checkNumFields( String str, int expected ) 
		throws DMLRuntimeException 
	{
		//note: split required for empty tokens
		int numParts = str.split(Instruction.OPERAND_DELIM).length;
		int numFields = numParts - 2; // -2 accounts for execType and opcode
		
		if ( numFields != expected ) 
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected + ") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}
	
	/**
	 * 
	 * @param str
	 * @param expected
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static int checkNumFields( String[] parts, int expected ) 
		throws DMLRuntimeException 
	{
		int numParts = parts.length;
		int numFields = numParts - 1; //account for opcode
		
		if ( numFields != expected ) 
			throw new DMLRuntimeException("checkNumFields() -- expected number (" + expected + ") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}
	
	/**
	 * 
	 * @param str
	 * @param expected1
	 * @param expected2
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static int checkNumFields( String str, int expected1, int expected2 ) 
		throws DMLRuntimeException 
	{
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
	 * @param str
	 * @return 
	 */
	public static String[] getInstructionParts( String str ) 
	{
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
	 * @param str
	 * @return
	 */
	public static String[] getInstructionPartsWithValueType( String str ) 
	{
		//note: split required for empty tokens
		String[] parts = str.split(Instruction.OPERAND_DELIM, -1);
		String[] ret = new String[parts.length-1]; // stripping-off the exectype
		ret[0] = parts[1]; // opcode
		for( int i=1; i<parts.length; i++ )
			ret[i-1] = parts[i];
		
		return ret;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 */
	public static String getOpCode( String str ) 
	{
		int ix1 = str.indexOf(Instruction.OPERAND_DELIM);
		int ix2 = str.indexOf(Instruction.OPERAND_DELIM, ix1+1);
		return str.substring(ix1+1, ix2);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLUnsupportedOperationException
	 */
	public static MRINSTRUCTION_TYPE getMRType( String str ) 
		throws DMLUnsupportedOperationException 
	{
		String opcode = getOpCode(str);
		MRINSTRUCTION_TYPE mrtype = MRInstructionParser.String2MRInstructionType.get( opcode ); 
		return mrtype;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLUnsupportedOperationException
	 */
	public static SPINSTRUCTION_TYPE getSPType( String str ) 
		throws DMLUnsupportedOperationException 
	{
		String opcode = getOpCode(str);
		SPINSTRUCTION_TYPE sptype = SPInstructionParser.String2SPInstructionType.get( opcode ); 
		return sptype;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLUnsupportedOperationException
	 */
	public static CPINSTRUCTION_TYPE getCPType( String str ) 
		throws DMLUnsupportedOperationException 
	{
		String opcode = getOpCode(str);
		CPINSTRUCTION_TYPE cptype = CPInstructionParser.String2CPInstructionType.get( opcode ); 
		return cptype;
	}
	
	/**
	 * 
	 * @param opcode
	 * @return
	 */
	public static boolean isBuiltinFunction ( String opcode ) 
	{
		Builtin.BuiltinFunctionCode bfc = Builtin.String2BuiltinFunctionCode.get(opcode);
		return (bfc != null);
	}
	
	public static boolean isOperand(String str) 
	{
		//note: split required for empty tokens
		String[] parts = str.split(Instruction.DATATYPE_PREFIX);
		return (parts.length > 1);
	}
	
	/**
	 * Evaluates if at least one instruction of the given instruction set
	 * used the distributed cache.
	 * 
	 * @param str
	 * @return
	 */
	public static boolean isDistributedCacheUsed(String str) 
	{	
		String[] parts = str.split(Instruction.INSTRUCTION_DELIM);
		for(String inst : parts) 
		{
			String opcode = getOpCode(inst);
			if(  opcode.equalsIgnoreCase(AppendM.OPCODE)  
			   || opcode.equalsIgnoreCase(MapMult.OPCODE)
			   || opcode.equalsIgnoreCase(PMMJ.OPCODE)
			   || opcode.equalsIgnoreCase(MapMultChain.OPCODE)
			   || opcode.equalsIgnoreCase(WeightedSquaredLoss.OPCODE)
			   || opcode.equalsIgnoreCase(WeightedSquaredLossR.OPCODE)
			   || opcode.equalsIgnoreCase(WeightedSigmoid.OPCODE)
			   || opcode.equalsIgnoreCase(WeightedSigmoidR.OPCODE)
			   || opcode.equalsIgnoreCase(UAggOuterChain.OPCODE)
			   || BinaryM.isOpcode( opcode ) ) //multiple opcodes	
			{
				return true;
			}
		}
		return false;
	}
	
	/**
	 * 
	 * @param opcode
	 * @return
	 */
	public static AggregateUnaryOperator parseBasicAggregateUnaryOperator(String opcode)
	{
		AggregateUnaryOperator aggun = null;
		
		if ( opcode.equalsIgnoreCase("uak+") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
		} 		
		else if ( opcode.equalsIgnoreCase("uark+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uack+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTROW);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		else if ( opcode.equalsIgnoreCase("uamean") ) {
			// Mean
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uarmean") ) {
			// RowMeans
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uacmean") ) {
			// ColMeans
			AggregateOperator agg = new AggregateOperator(0, Mean.getMeanFnObject(), true, CorrectionLocationType.LASTTWOROWS);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		else if ( opcode.equalsIgnoreCase("ua+") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uar+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uac+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		else if ( opcode.equalsIgnoreCase("ua*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uamax") ) {
			AggregateOperator agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
		}
		else if ( opcode.equalsIgnoreCase("uamin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uatrace") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uaktrace") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject());
		} 		
		else if ( opcode.equalsIgnoreCase("uarmax") ) {
			AggregateOperator agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		} 
		else if (opcode.equalsIgnoreCase("uarimax") ) {
			AggregateOperator agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		}
		else if ( opcode.equalsIgnoreCase("uarmin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		} 
		else if (opcode.equalsIgnoreCase("uarimin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("minindex"), true, CorrectionLocationType.LASTCOLUMN);
			aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
		}
		else if ( opcode.equalsIgnoreCase("uacmax") ) {
			AggregateOperator agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("max"));
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("uacmin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		
		return aggun;
	}
	
	/**
	 * 
	 * @param opcode
	 * @param corrExists
	 * @param corrLoc
	 * @return
	 */
	public static AggregateOperator parseAggregateOperator(String opcode, String corrExists, String corrLoc)
	{
		AggregateOperator agg = null;
	
		if ( opcode.equalsIgnoreCase("ak+") || opcode.equalsIgnoreCase("aktrace") ) {
			boolean lcorrExists = (corrExists==null) ? true : Boolean.parseBoolean(corrExists);
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTCOLUMN : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), lcorrExists, lcorrLoc);
		} 
		else if ( opcode.equalsIgnoreCase("a+") ) {
			agg = new AggregateOperator(0, Plus.getPlusFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("a*") ) {
			agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
		}
		else if (opcode.equalsIgnoreCase("arimax")){
			agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
		}
		else if ( opcode.equalsIgnoreCase("amax") ) {
			agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("max"));
		}
		else if ( opcode.equalsIgnoreCase("amin") ) {
			agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
		}
		else if (opcode.equalsIgnoreCase("arimin")){
			agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("minindex"), true, CorrectionLocationType.LASTCOLUMN);
		}
		else if ( opcode.equalsIgnoreCase("amean") ) {
			boolean lcorrExists = (corrExists==null) ? true : Boolean.parseBoolean(corrExists);
			CorrectionLocationType lcorrLoc = (corrLoc==null) ? CorrectionLocationType.LASTTWOCOLUMNS : CorrectionLocationType.valueOf(corrLoc);
			agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), lcorrExists, lcorrLoc);
		}
		
		return agg;
	}
	
	/**
	 * 
	 * @param opcode
	 * @return
	 */
	public static AggregateUnaryOperator parseCumulativeAggregateUnaryOperator(String opcode)
	{
		AggregateUnaryOperator aggun = null;
		if( "ucumack+".equals(opcode) ) { 
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTROW);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		else if ( "ucumac*".equals(opcode) ) { 
			AggregateOperator agg = new AggregateOperator(0, Multiply.getMultiplyFnObject(), false, CorrectionLocationType.NONE);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		else if ( "ucumacmin".equals(opcode) ) { 
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"), false, CorrectionLocationType.NONE);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		else if ( "ucumacmax".equals(opcode) ) { 
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"), false, CorrectionLocationType.NONE);
			aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
		}
		
		return aggun;
	}
	
	/**
	 * 
	 * @param opcode
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static BinaryOperator parseBinaryOperator(String opcode) 
		throws DMLRuntimeException
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
		else if(opcode.equalsIgnoreCase("+"))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase("-"))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase("*"))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if ( opcode.equalsIgnoreCase("*2") ) 
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase("/"))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase("%%"))
			return new BinaryOperator(Modulus.getModulusFnObject());
		else if(opcode.equalsIgnoreCase("%/%"))
			return new BinaryOperator(IntegerDivide.getIntegerDivideFnObject());
		else if(opcode.equalsIgnoreCase("^"))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase("^2") )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase("max") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase("min") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		
		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}
	
	/**
	 * scalar-matrix operator 
	 * @param opcode
	 * @param arg1IsScalar
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static ScalarOperator parseScalarBinaryOperator(String opcode, boolean arg1IsScalar)
		throws DMLRuntimeException
	{
		double default_constant = 0;
		
		//commutative operators
		if ( opcode.equalsIgnoreCase("+") ){ 
			return new RightScalarOperator(Plus.getPlusFnObject(), default_constant); 
		}
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new RightScalarOperator(Multiply.getMultiplyFnObject(), default_constant);
		} 
		//non-commutative operators but both scalar-matrix and matrix-scalar makes sense
		else if ( opcode.equalsIgnoreCase("-") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Minus.getMinusFnObject(), default_constant);
			else return new RightScalarOperator(Minus.getMinusFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("-nz") ) {
			//no support for left scalar yet
			return new RightScalarOperator(MinusNz.getMinusNzFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("/") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Divide.getDivideFnObject(), default_constant);
			else return new RightScalarOperator(Divide.getDivideFnObject(), default_constant);
		}  
		else if ( opcode.equalsIgnoreCase("%%") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Modulus.getModulusFnObject(), default_constant);
			else return new RightScalarOperator(Modulus.getModulusFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("%/%") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(IntegerDivide.getIntegerDivideFnObject(), default_constant);
			else return new RightScalarOperator(IntegerDivide.getIntegerDivideFnObject(), default_constant);
		}
		//operations for which only matrix-scalar makes sense
		else if ( opcode.equalsIgnoreCase("^") ){
			if(arg1IsScalar)
				return new LeftScalarOperator(Power.getPowerFnObject(), default_constant);
			else return new RightScalarOperator(Power.getPowerFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("max"), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("min") ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("min"), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("log") ){
			return new RightScalarOperator(Builtin.getBuiltinFnObject("log"), default_constant);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(GreaterThan.getGreaterThanFnObject(), default_constant);
			return new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), default_constant);
			return new RightScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(LessThan.getLessThanFnObject(), default_constant);
			return new RightScalarOperator(LessThan.getLessThanFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), default_constant);
			return new RightScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Equals.getEqualsFnObject(), default_constant);
			return new RightScalarOperator(Equals.getEqualsFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(NotEquals.getNotEqualsFnObject(), default_constant);
			return new RightScalarOperator(NotEquals.getNotEqualsFnObject(), default_constant);
		}
		
		//operation that only exist for performance purposes
		else if ( opcode.equalsIgnoreCase("*2") ) {
			return new RightScalarOperator(Multiply2.getMultiply2FnObject(), default_constant);
		} 
		else if ( opcode.equalsIgnoreCase("^2") ){
			return new RightScalarOperator(Power2.getPower2FnObject(), default_constant);
		}
		
		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}	


	/**
	 * 
	 * @param opcode
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static BinaryOperator parseExtendedBinaryOperator(String opcode) 
		throws DMLRuntimeException
	{
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
		else if(opcode.equalsIgnoreCase("&&"))
			return new BinaryOperator(And.getAndFnObject());
		else if(opcode.equalsIgnoreCase("||"))
			return new BinaryOperator(Or.getOrFnObject());
		else if(opcode.equalsIgnoreCase("+") || opcode.equalsIgnoreCase("map+"))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase("-") || opcode.equalsIgnoreCase("map-"))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase("*") || opcode.equalsIgnoreCase("map*"))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if ( opcode.equalsIgnoreCase("*2") ) 
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase("/") || opcode.equalsIgnoreCase("map/"))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase("%%") || opcode.equalsIgnoreCase("map%%"))
			return new BinaryOperator(Modulus.getModulusFnObject());
		else if(opcode.equalsIgnoreCase("%/%") || opcode.equalsIgnoreCase("map%/%"))
			return new BinaryOperator(IntegerDivide.getIntegerDivideFnObject());
		else if(opcode.equalsIgnoreCase("^") || opcode.equalsIgnoreCase("map^"))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase("^2") )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase("max") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase("min") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		
		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}
	
	
	/**
	 * 
	 * @param opcode
	 * @return
	 */
	public static String deriveAggregateOperatorOpcode(String opcode)
	{
		if ( opcode.equalsIgnoreCase("uak+") || opcode.equalsIgnoreCase("uark+") || opcode.equalsIgnoreCase("uack+"))
			return "ak+";
		else if ( opcode.equalsIgnoreCase("uamean") || opcode.equalsIgnoreCase("uarmean") || opcode.equalsIgnoreCase("uacmean") )
			return "amean";
		else if ( opcode.equalsIgnoreCase("ua+") || opcode.equalsIgnoreCase("uar+") || opcode.equalsIgnoreCase("uac+") )
			return "a+";
		else if ( opcode.equalsIgnoreCase("ua*") )
			return "a*";
		else if ( opcode.equalsIgnoreCase("uatrace") || opcode.equalsIgnoreCase("uaktrace") ) 
			return "aktrace";
		else if ( opcode.equalsIgnoreCase("uamax") || opcode.equalsIgnoreCase("uarmax") || opcode.equalsIgnoreCase("uacmax") )
			return "amax";
		else if ( opcode.equalsIgnoreCase("uamin") || opcode.equalsIgnoreCase("uarmin") || opcode.equalsIgnoreCase("uacmin") )
			return "amin";
		else if (opcode.equalsIgnoreCase("uarimax") )
			return "arimax";
		else if (opcode.equalsIgnoreCase("uarimin") )
			return "arimin";
	
		return null;
	}

	/**
	 * 
	 * @param opcode
	 * @return
	 */
	public static CorrectionLocationType deriveAggregateOperatorCorrectionLocation(String opcode)
	{
		if (   opcode.equalsIgnoreCase("uak+") || opcode.equalsIgnoreCase("uark+") || opcode.equalsIgnoreCase("uatrace") || opcode.equalsIgnoreCase("uaktrace") )
			return CorrectionLocationType.LASTCOLUMN;
		else if ( opcode.equalsIgnoreCase("uack+") ) 
			return CorrectionLocationType.LASTROW;
		else if ( opcode.equalsIgnoreCase("uamean") || opcode.equalsIgnoreCase("uarmean") )
			return CorrectionLocationType.LASTTWOCOLUMNS;
		else if ( opcode.equalsIgnoreCase("uacmean") )
			return CorrectionLocationType.LASTTWOROWS;
		else if (opcode.equalsIgnoreCase("uarimax") || opcode.equalsIgnoreCase("uarimin") )	
			return CorrectionLocationType.LASTCOLUMN;
		
		return CorrectionLocationType.NONE;
	}
}
