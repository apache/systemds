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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;


/**
 * TODO Currently this lop only support the right hand side in distributed cache. This
 *  should be generalized (incl hop operator selection) to left/right cache types.
 *  
 * 
 */
public class UAggOuterChain extends Lop 
{
	public static final String OPCODE = Opcodes.UAGGOUTERCHAIN.toString();

	//outer operation
	private AggOp _uaggOp = null;
	private Direction _uaggDir = null;
	//inner operation
	private OpOp2 _binOp = null;
	
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
	public UAggOuterChain(Lop input1, Lop input2, AggOp uaop, Direction uadir, OpOp2 bop, DataType dt, ValueType vt, ExecType et) {
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
			getExecType().name(), OPCODE,
			PartialAggregate.getOpcode(_uaggOp, _uaggDir), //outer
			_binOp.toString(), //inner
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(0).prepInputOperand(input2),
			prepOutputOperand(output));
	}
}
