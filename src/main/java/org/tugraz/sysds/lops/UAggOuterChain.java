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


/**
 * TODO Currently this lop only support the right hand side in distributed cache. This
 *  should be generalized (incl hop operator selection) to left/right cache types.
 *  
 * 
 */
public class UAggOuterChain extends Lop 
{
	
	public static final String OPCODE = "uaggouterchain";

	//outer operation
	private Aggregate.OperationTypes _uaggOp         = null;
	private PartialAggregate.DirectionTypes _uaggDir = null;
	//inner operation
	private Binary.OperationTypes _binOp             = null;	
		
	
	/**
	 * Constructor to setup a unaryagg outer chain
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param uaop aggregate operation type
	 * @param uadir partial aggregate direction type
	 * @param bop binary operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public UAggOuterChain(Lop input1, Lop input2, Aggregate.OperationTypes uaop, PartialAggregate.DirectionTypes uadir, Binary.OperationTypes bop, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UaggOuterChain, dt, vt);
		addInput(input1);
		addInput(input2);
		input1.addOutput(this); 
		input2.addOutput(this); 
		
		//setup operator types
		_uaggOp = uaop;
		_uaggDir = uadir;
		_binOp = bop;
		lps.setProperties(inputs, et);
	}
	
	@Override
	public String toString() {
		return "Operation = UaggOuterChain";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			PartialAggregate.getOpcode(_uaggOp, _uaggDir), //outer
			Binary.getOpcode(_binOp), //inner
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(0).prepInputOperand(input2),
			prepOutputOperand(output));
	}
}
