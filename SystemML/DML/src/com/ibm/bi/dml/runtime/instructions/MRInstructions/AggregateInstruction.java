/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AggregateInstruction extends UnaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public AggregateInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Aggregate;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		if(opcode.equalsIgnoreCase("ak+") || opcode.equalsIgnoreCase("amean"))
			InstructionUtils.checkNumFields ( str, 4 );
		else
			InstructionUtils.checkNumFields ( str, 2 );
		
		if ( opcode.equalsIgnoreCase("ak+") ) {
			boolean corExists=Boolean.parseBoolean(parts[3]);
			CorrectionLocationType loc;
			try {
				loc = PartialAggregate.decodeCorrectionLocation(parts[4]);
			} catch (LopsException e) {
				throw new DMLRuntimeException(e);
			}
			
			// if corrections are not available, then we must use simple sum
			AggregateOperator agg = null; 
			if ( corExists ) {
				agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corExists, loc);
			}
			else {
				agg = new AggregateOperator(0, Plus.getPlusFnObject(), corExists, loc);
			}
			return new AggregateInstruction(agg, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("a+") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			return new AggregateInstruction(agg, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("a*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			return new AggregateInstruction(agg, in, out, str);
		}
		else if (opcode.equalsIgnoreCase("arimax")){
			AggregateOperator agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
			return new AggregateInstruction(agg, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("amax") ) {
			AggregateOperator agg = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("max"));
			return new AggregateInstruction(agg, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("amin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			return new AggregateInstruction(agg, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("amean") ) {
			boolean corExists=Boolean.parseBoolean(parts[3]);
			CorrectionLocationType loc;
			try {
				loc = PartialAggregate.decodeCorrectionLocation(parts[4]);
			} catch (LopsException e) {
				throw new DMLRuntimeException(e);
			}
			
			// if corrections are not available, then we must use simple sum
			AggregateOperator agg = null; 
			if ( corExists ) {
				// stable mean internally makes use of Kahan summation
				agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corExists, loc);
			}
			else {
				agg = new AggregateOperator(0, Plus.getPlusFnObject(), corExists, loc);
			}
			return new AggregateInstruction(agg, in, out, str);
		}
		return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		throw new DMLRuntimeException("no processInstruction for AggregateInstruction!");
		
	}

}
