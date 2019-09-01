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

package org.tugraz.sysds.lops;

import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;


public class BinaryUAggChain extends Lop 
{
	
	public static final String OPCODE = "binuaggchain";

	//outer operation
	private Binary.OperationTypes _binOp             = null;	
	//inner operation
	private Aggregate.OperationTypes _uaggOp         = null;
	private PartialAggregate.DirectionTypes _uaggDir = null;
	
	
	/**
	 * Constructor to setup a map mult chain without weights
	 * 
	 * 
	 * @param input1 low-level operator
	 * @param bop binary operation type
	 * @param uaop aggregate operation type
	 * @param uadir partial aggregate direction type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public BinaryUAggChain(Lop input1, Binary.OperationTypes bop, Aggregate.OperationTypes uaop, PartialAggregate.DirectionTypes uadir, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.BinUaggChain, dt, vt);
		addInput(input1); //X
		input1.addOutput(this); 
		
		//setup operator types
		_binOp = bop;
		_uaggOp = uaop;
		_uaggDir = uadir;
		lps.setProperties(inputs, et);
	}
	
	@Override
	public String toString() {
		return "Operation = BinUaggChain";
	}
	
	@Override
	public String getInstructions(String input1, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			Binary.getOpcode(_binOp), //outer opcode
			PartialAggregate.getOpcode(_uaggOp, _uaggDir), //inner opcode
			getInputs().get(0).prepInputOperand(input1),
			prepOutputOperand(output));
	}
}
