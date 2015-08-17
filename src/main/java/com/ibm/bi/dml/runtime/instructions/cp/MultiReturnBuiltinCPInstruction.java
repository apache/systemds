/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.util.ArrayList;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.LibCommonsMath;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class MultiReturnBuiltinCPInstruction extends ComputationCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	int arity;
	protected ArrayList<CPOperand> _outputs;
	
	public MultiReturnBuiltinCPInstruction(Operator op, CPOperand input1, ArrayList<CPOperand> outputs, String opcode, String istr )
	{
		super(op, input1, null, outputs.get(0), opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MultiReturnBuiltin;
		_outputs = outputs;
	}

	public int getArity() {
		return arity;
	}
	
	public CPOperand getOutput(int i)
	{
		return _outputs.get(i);
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<CPOperand>();
		// first part is always the opcode
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("qr") ) {
			// one input and two ouputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("lu") ) {
			CPOperand in1 = new CPOperand(parts[1]);
			
			// one input and three outputs
			outputs.add ( new CPOperand(parts[2], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
			
		}
		else if ( opcode.equalsIgnoreCase("eigen") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
			
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String opcode = getOpcode();
		MatrixObject mo = (MatrixObject) ec.getVariable(input1.getName());
		MatrixBlock[] out = null;
		
		if(LibCommonsMath.isSupportedMultiReturnOperation(opcode))
			out = LibCommonsMath.multiReturnOperations(mo, opcode);
		else 
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);

		
		for(int i=0; i < _outputs.size(); i++) {
			ec.setMatrixOutput(_outputs.get(i).getName(), out[i]);
		}
	}
}
