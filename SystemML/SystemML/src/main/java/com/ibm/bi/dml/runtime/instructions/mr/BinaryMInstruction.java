/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.AppendM.CacheType;
import com.ibm.bi.dml.lops.BinaryM.VectorType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.IntegerDivide;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class BinaryMInstruction extends BinaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CacheType _cacheType = null;
	private VectorType _vectorType = null;
	
	public BinaryMInstruction(Operator op, byte in1, byte in2, CacheType ctype, VectorType vtype, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
		instString = istr;
		
		_cacheType = ctype;
		_vectorType = vtype;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		CacheType ctype = CacheType.valueOf(parts[4]);
		VectorType vtype = VectorType.valueOf(parts[5]);
		
		if ( opcode.equalsIgnoreCase("map+") ) {
			return new BinaryMInstruction(new BinaryOperator(Plus.getPlusFnObject()), in1, in2, ctype, vtype, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("map-") ) {
			return new BinaryMInstruction(new BinaryOperator(Minus.getMinusFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map*") ) {
			return new BinaryMInstruction(new BinaryOperator(Multiply.getMultiplyFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map/") ) {
			return new BinaryMInstruction(new BinaryOperator(Divide.getDivideFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map^") ) {
			return new BinaryMInstruction(new BinaryOperator(Power.getPowerFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map%%") ) {
			return new BinaryMInstruction(new BinaryOperator(Modulus.getModulusFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map%/%") ) {
			return new BinaryMInstruction(new BinaryOperator(IntegerDivide.getIntegerDivideFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("mapmin") ) {
			return new BinaryMInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("min")), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("mapmax") ) {
			return new BinaryMInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("max")), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map>") ) {
			return new BinaryMInstruction(new BinaryOperator(GreaterThan.getGreaterThanFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map>=") ) {
			return new BinaryMInstruction(new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map<") ) {
			return new BinaryMInstruction(new BinaryOperator(LessThan.getLessThanFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map<=") ) {
			return new BinaryMInstruction(new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map==") ) {
			return new BinaryMInstruction(new BinaryOperator(Equals.getEqualsFnObject()), in1, in2, ctype, vtype, out, str);
		}
		else if ( opcode.equalsIgnoreCase("map!=") ) {
			return new BinaryMInstruction(new BinaryOperator(NotEquals.getNotEqualsFnObject()), in1, in2, ctype, vtype, out, str);
		}
		return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.getFirst(input1);
		if( in1==null )
			return;
		
		//allocate space for the output value
		//try to avoid coping as much as possible
		IndexedMatrixValue out;
		if( (output!=input1 && output!=input2) )
			out=cachedValues.holdPlace(output, valueClass);
		else
			out=tempValue;
		
		//get second 
		DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
		IndexedMatrixValue in2 = null;
		if( _vectorType == VectorType.COL_VECTOR )
			in2 = dcInput.getDataBlock((int)in1.getIndexes().getRowIndex(), 1);
		else //_vectorType == VectorType.ROW_VECTOR
			in2 = dcInput.getDataBlock(1, (int)in1.getIndexes().getColumnIndex());
		
		//process instruction
		out.getIndexes().setIndexes(in1.getIndexes());
		OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1.getValue(), 
				in2.getValue(), out.getValue(), ((BinaryOperator)optr));
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
	}

	public static boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		boolean ret = false;
		
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		ret = (index==in2 && index!=in1);
		
		return ret;
	}
	
	public static void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		indexes.add(in2);
	}
}
