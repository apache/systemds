/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;


public class AppendCPInstruction extends BinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum AppendType{
		CBIND,
		STRING,
	}
	
	//offset of second matrix, can be a var name or constant
	private CPOperand _offset;
	
	//type (matrix cbind / scalar string concatenation)
	private AppendType _type;
	
	public AppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String istr, AppendType type)
	{
		super(op, in1, in2, out, istr);
		cptype = CPINSTRUCTION_TYPE.Append;
		
		_offset = in3;
		_type = type;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and offset = 4
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		//String offset_str = parts[4];
		 
		AppendType type = (in1.get_dataType()==DataType.MATRIX) ? AppendType.CBIND : AppendType.STRING;
		
		
		if(!opcode.equalsIgnoreCase("append"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);
		else
			return new AppendCPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, in3, out, str, type);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if( _type == AppendType.CBIND )
		{
			//get inputs
			MatrixBlock matBlock1 = ec.getMatrixInput(input1.get_name());
			MatrixBlock matBlock2 = ec.getMatrixInput(input2.get_name());
			
			//check input dimensions
			if(matBlock1.getNumRows() != matBlock2.getNumRows())
				throw new DMLRuntimeException("Append is not possible for input matrices " 
											  + input1.get_name() + " and " + input2.get_name()
											  + "with unequal number of rows");
			
			//execute append operations (append both inputs to initially empty output)
			MatrixBlock ret = matBlock1.appendOperations(matBlock2, new MatrixBlock());
			
			//set output
			ec.setMatrixOutput(output.get_name(), ret);
			
			//release inputs 
			ec.releaseMatrixInput(input1.get_name());
			ec.releaseMatrixInput(input2.get_name());
		}
		else //STRING
		{
			//get input strings (vars or literals)
			ScalarObject so1 = ec.getScalarInput( input1.get_name(), input1.get_valueType(), input1.isLiteral() );
			ScalarObject so2 = ec.getScalarInput( input2.get_name(), input2.get_valueType(), input2.isLiteral() );
			
			//pre-checks
			String val1 = so1.getStringValue();
			String val2 = so2.getStringValue();
			StringObject.checkMaxStringLength( val1.length()+val2.length() );
			
			//core execution
			String outString = val1 + "\n" + val2;			
			ScalarObject sores = new StringObject(outString);
			
			//set output
			ec.setScalarOutput(output.get_name(), sores);
		}		
	}
}
