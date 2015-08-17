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

package com.ibm.bi.dml.runtime.instructions.cp;

import java.io.IOException;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.RemoveFile;
import com.ibm.bi.dml.runtime.functionobjects.RenameFile;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;



public class FileCPInstruction extends CPInstruction 
{
	
	private enum FileOperationCode {
		RemoveFile, MoveFile
	};

	private FileOperationCode _code;
	private String _input1;
	private String _input2;
	//private int _arity;
	
	public FileCPInstruction (Operator op, FileOperationCode code, String in1, String in2, int arity, String opcode, String istr )
	{
		super(op, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.File;
		_code = code;
		_input1 = in1;
		_input2 = in2;
		//_arity = arity;
	}

	private static FileOperationCode getFileOperationCode ( String str ) 
		throws DMLUnsupportedOperationException 
	{
		if ( str.equalsIgnoreCase("rm"))
			return FileOperationCode.RemoveFile;
		else if ( str.equalsIgnoreCase("mv") ) 
			return FileOperationCode.MoveFile;
		else
			throw new DMLUnsupportedOperationException("Invalid function: " + str);
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		
		int _arity = 2;
		if ( opcode.equalsIgnoreCase("rm") ) 
			_arity = 1;
		
		InstructionUtils.checkNumFields ( str, _arity ); // there is no output, so we just have _arity
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		
		String in1, in2;
		opcode = parts[0];
		FileOperationCode focode = getFileOperationCode(opcode); 
		in1 = parts[1];
		in2 = null;
		if ( _arity == 2 )
			in2 = parts[2];
		
		// Determine appropriate Function Object based on opcode
		
		if ( opcode.equalsIgnoreCase("rm") ) {
			return new FileCPInstruction(new SimpleOperator(RemoveFile.getRemoveFileFnObject()), focode, in1, in2, _arity, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("mv") ) {
			return new FileCPInstruction(new SimpleOperator(RenameFile.getRenameFileFnObject()), focode, in1, in2, _arity, opcode, str);
		}
		return null;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{	
		try 
		{
			switch(_code) 
			{
				case RemoveFile:
					MapReduceTool.deleteFileIfExistOnHDFS(_input1);
					MapReduceTool.deleteFileIfExistOnHDFS(_input1+".mtd");
					break;
				case MoveFile:
					MapReduceTool.renameFileOnHDFS(_input1, _input2);
					MapReduceTool.renameFileOnHDFS(_input1+".mtd", _input2+".mtd");
					break;
					
				default:
					throw new DMLRuntimeException("Unexpected opcode: " + _code);
			}
		} 
		catch ( IOException e ) {
			throw new DMLRuntimeException(e);
		}
		
		// NO RESULT is produced
		// pb.setVariable(output.getName(), sores);
	}
	
	

}
