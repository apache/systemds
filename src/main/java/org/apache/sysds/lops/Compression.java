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

import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

public class Compression extends Lop 
{
	public static final String OPCODE = "compress";
	
	public enum CompressConfig {
		TRUE,
		FALSE,
		AUTO;
		public boolean isEnabled() {
			return this == TRUE || this == AUTO;
		}
	}
	
	public Compression(Lop input, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.Checkpoint, dt, vt);
		addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		return "Compress";
	}
	
	@Override
	public String getInstructions(String input1, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		return sb.toString();
	}
}
