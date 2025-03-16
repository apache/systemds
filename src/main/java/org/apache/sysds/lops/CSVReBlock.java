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
 
package org.apache.sysds.lops;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.instructions.InstructionUtils;


/**
 * Lop to convert CSV data into SystemDS data format
 */
public class CSVReBlock extends Lop 
{
	public static final String OPCODE = Opcodes.CSVRBLK.toString();
	
	private int _blocksize;

	public CSVReBlock(Lop input, int blen, DataType dt, ValueType vt, ExecType et)
	{
		super(Lop.Type.CSVReBlock, dt, vt);
		addInput(input);
		input.addOutput(this);
		
		_blocksize = blen;
		
		if(et == ExecType.SPARK) {
			lps.setProperties( inputs, ExecType.SPARK);
		}
		else {
			throw new LopsException("Incorrect execution type for CSVReblock:" + et);
		}
	}

	@Override
	public String toString() {
		return "CSVReblock - blocksize = " + _blocksize;
	}
	
	private String prepCSVProperties() {
		StringBuilder sb = new StringBuilder();

		Data dataInput = (Data)getInputs().get(0);
		
		Lop headerLop = dataInput.getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW, 
			String.valueOf(DataExpression.DEFAULT_DELIM_HAS_HEADER_ROW));
		Lop delimLop = dataInput.getNamedInputLop(DataExpression.DELIM_DELIMITER, 
			DataExpression.DEFAULT_DELIM_DELIMITER);
		Lop fillLop = dataInput.getNamedInputLop(DataExpression.DELIM_FILL, 
			String.valueOf(DataExpression.DEFAULT_DELIM_FILL)); 
		Lop fillValueLop = dataInput.getNamedInputLop(DataExpression.DELIM_FILL_VALUE, 
			String.valueOf(DataExpression.DEFAULT_DELIM_FILL_VALUE));
		Lop naStrings = dataInput.getNamedInputLop(DataExpression.DELIM_NA_STRINGS,
			String.valueOf(DataExpression.DEFAULT_NA_STRINGS));
		
		if (headerLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
					+ " must be a literal.");
		if (delimLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_DELIMITER
					+ " must be a literal.");
		if (fillLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_FILL
					+ " must be a literal.");
		if (fillValueLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_FILL_VALUE
					+ " must be a literal.");

		sb.append( ((Data)headerLop).getBooleanValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)delimLop).getStringValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)fillLop).getBooleanValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)fillValueLop).getDoubleValue() );
		sb.append( OPERAND_DELIMITOR );
		if(naStrings instanceof Nary){
			Nary naLops = (Nary) naStrings;
			for(Lop na : naLops.getInputs()){
				sb.append(((Data)na).getStringValue());
				sb.append(DataExpression.DELIM_NA_STRING_SEP);
			}
		} else if (naStrings instanceof Data){
			sb.append(((Data)naStrings).getStringValue());
		}
		
		return sb.toString();
	}

	@Override
	public String getInstructions(String input1, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			getInputs().get(0).prepInputOperand(input1),
			prepOutputOperand(output),
			String.valueOf(_blocksize),
			prepCSVProperties());
	}
}
