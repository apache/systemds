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

package org.apache.sysml.runtime.instructions.mr;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;


/**
 * Supported opcodes: replace.
 * 
 */
public class ParameterizedBuiltinMRInstruction extends UnaryInstruction
{	
	String _opcode = null;
	
	//replace-specific attributes
	private double _pattern;
	private double _replace;
	
	//rexpand-specific attributes
	private double _max;
	private boolean _dirRows;
	private boolean _cast;
	private boolean _ignore;
	
	public ParameterizedBuiltinMRInstruction(Operator op, byte in, double pattern, double replace, byte out, String opcode, String istr)
	{
		super(op, in, out, istr);
		instString = istr;
		_opcode = opcode;
		_pattern = pattern;
		_replace = replace;
	}
	
	public ParameterizedBuiltinMRInstruction(Operator op, byte in, double max, boolean dirRows, boolean cast, boolean ignore, byte out, String opcode, String istr)
	{
		super(op, in, out, istr);
		instString = istr;
		_opcode = opcode;
		_max = max;
		_dirRows = dirRows;
		_cast = cast;
		_ignore = ignore;
	}
	
	/**
	 * 
	 * @param mcIn
	 * @param mcOut
	 */
	public void computeOutputCharacteristics(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut)
	{
		if( _opcode.equalsIgnoreCase("replace") ) {
			mcOut.set(mcIn);
		}
		else if( _opcode.equalsIgnoreCase("rexpand") )
		{
			long lmax = UtilFunctions.toLong(_max);
			if( _dirRows )
				mcOut.set(lmax, mcIn.getRows(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
			else
				mcOut.set(mcIn.getRows(), lmax, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());	
		}
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		
		if( opcode.equalsIgnoreCase("replace") )
		{
			InstructionUtils.checkNumFields ( str, 4 );
			
			byte in = Byte.parseByte(parts[1]);
			double pattern = Double.parseDouble(parts[2]);
			double replace = Double.parseDouble(parts[3]);
			byte out = Byte.parseByte(parts[4]);
			
			return new ParameterizedBuiltinMRInstruction(new Operator(true), in, pattern, replace, out, opcode, str);
		}
		else if( opcode.equalsIgnoreCase("rexpand") )
		{
			InstructionUtils.checkNumFields ( str, 6 );
			
			byte in = Byte.parseByte(parts[1]);
			double max = Double.parseDouble(parts[2]);
			boolean dirRows = parts[3].equals("rows");
			boolean cast = Boolean.parseBoolean(parts[4]);
			boolean ignore = Boolean.parseBoolean(parts[5]);
			byte out = Byte.parseByte(parts[6]);
			
			return new ParameterizedBuiltinMRInstruction(new Operator(true), in, max, dirRows, cast, ignore, out, opcode, str);			
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing an ParameterizedBuiltinMRInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList !=null )
			for(IndexedMatrixValue imv : blkList)
			{
				if(imv==null)
					continue;
				
				if( _opcode.equalsIgnoreCase("replace") )
				{
					MatrixValue in = imv.getValue();
					MatrixIndexes inIX = imv.getIndexes();
					
					//allocate space for the output value
					IndexedMatrixValue iout = null;
					if(output==input)
						iout=tempValue;
					else
						iout=cachedValues.holdPlace(output, valueClass);
					iout.getIndexes().setIndexes(inIX);
					MatrixValue out = iout.getValue();
					
					//process instruction
					in.replaceOperations(out, _pattern, _replace);
					
					//put the output value in the cache
					if(iout==tempValue)
						cachedValues.add(output, iout);
				}
				else if( _opcode.equalsIgnoreCase("rexpand") )
				{
					//process instruction
					ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();
					LibMatrixReorg.rexpand(imv, _max, _dirRows, _cast, _ignore, blockRowFactor, blockColFactor, out);
					
					//put the output values in the cache
					for( IndexedMatrixValue lout : out )
						cachedValues.add(output, lout);
				}
			}
	}
}
