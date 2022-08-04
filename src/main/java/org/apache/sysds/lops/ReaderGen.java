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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.DataExpression;

import java.util.HashMap;

public class ReaderGen extends Lop
{
	public static final String OPCODE = "readergen";
	private HashMap<String, Lop> _inputParams;

	public ReaderGen(Lop input, DataType dt, ValueType vt, ExecType et, HashMap<String, Lop> inputParametersLops) {
		super(Type.Checkpoint, dt, vt);
		addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
		_inputParams = inputParametersLops;
	}

	@Override
	public String toString() {
		return "ReaderGen";
	}

	@Override
	public String getInstructions(){
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );

		// sample matrix/frame
		Lop inSample = _inputParams.get(DataExpression.SAMPLE);
		sb.append( OPERAND_DELIMITOR );
		sb.append ( inSample.prepInputOperand(inSample.getOutputParameters().getLabel()));

		// sample Raw
		Lop inSampleRaw = _inputParams.get(DataExpression.SAMPLE_RAW);
		sb.append( OPERAND_DELIMITOR );
		sb.append (inSampleRaw.getOutputParameters().getLabel());

		// src output
		Lop inOutput = _inputParams.get(DataExpression.FORMAT_TYPE);
		sb.append( OPERAND_DELIMITOR );
		sb.append(inOutput.getOutputParameters().getLabel());

		return sb.toString();
	}
}
