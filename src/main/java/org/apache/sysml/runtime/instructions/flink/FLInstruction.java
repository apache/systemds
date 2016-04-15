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

package org.apache.sysml.runtime.instructions.flink;

import org.apache.sysml.lops.runtime.RunMRJobs;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.FLInstructionParser;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.Statistics;

public abstract class FLInstruction extends Instruction {

	public enum FLINSTRUCTION_TYPE {
		TSMM, MAPMM, Reblock, CSVReblock, Write, INVALID
	}

	protected FLINSTRUCTION_TYPE _fltype;
	protected Operator _optr;

	protected boolean _requiresLabelUpdate = false;

	public FLInstruction(String opcode, String istr) {
		type = INSTRUCTION_TYPE.FLINK;
		instString = istr;
		instOpcode = opcode;

		//update requirement for repeated usage
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}

	public FLInstruction(Operator op, String opcode, String istr) {
		this(opcode, istr);
		_optr = op;
	}

	public FLINSTRUCTION_TYPE getFLInstructionType() {
		return _fltype;
	}

	@Override
	public boolean requiresLabelUpdate() {
		return _requiresLabelUpdate;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		//default pre-process behavior (e.g., debug state)
		Instruction tmp = super.preprocessInstruction(ec);

		//instruction patching
		if (tmp.requiresLabelUpdate()) //update labels only if required
		{
			//note: no exchange of updated instruction as labels might change in the general case
			String updInst = RunMRJobs.updateLabels(tmp.toString(), ec.getVariables());
			tmp = FLInstructionParser.parseSingleInstruction(updInst);
		}

		return tmp;
	}

	@Override
	public abstract void processInstruction(ExecutionContext ec) throws DMLRuntimeException;

	@Override
	public void postprocessInstruction(ExecutionContext ec) throws DMLRuntimeException {
		Statistics.incrementNoOfExecutedFLInst();

		super.postprocessInstruction(ec);
	}
}
