package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.BinaryM.VectorType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class RelationalBinarySPInstruction extends BinarySPInstruction {
	
	public RelationalBinarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.RelationalBinary;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = null;
		boolean isBroadcast = false;
		VectorType vtype = null;
		
		if(str.startsWith("SPARK"+Lop.OPERAND_DELIMITOR+"map")) {
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			InstructionUtils.checkNumFields ( parts, 5 );
			
			opcode = parts[0];
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			vtype = VectorType.valueOf(parts[5]);
			isBroadcast = true;
		}
		else {
			InstructionUtils.checkNumFields (str, 3);
			opcode = parseBinaryInstruction(str, in1, in2, out);
		}
		
		DataType dt1 = in1.getDataType();
		DataType dt2 = in2.getDataType();
		
		Operator operator = (dt1 != dt2) ?
					InstructionUtils.parseScalarBinaryOperator(opcode, (dt1 == DataType.SCALAR))
					: InstructionUtils.parseExtendedBinaryOperator(opcode);
		
		if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX){
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX) {
				if(isBroadcast)
					return new MatrixBVectorRelationalSPInstruction(operator, in1, in2, out, vtype, opcode, str);
				else
					return new MatrixMatrixRelationalSPInstruction(operator, in1, in2, out, opcode, str);
			}
			else
				return new MatrixScalarRelationalSPInstruction(operator, in1, in2, out, opcode, str);
		}
		
		return null;
	}
}
