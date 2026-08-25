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

package org.apache.sysds.runtime.instructions.cp;

import java.util.LinkedHashMap;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

/**
 * CP instruction for the dp_laplace/dp_gaussian opcodes. Subclasses {@link ParameterizedBuiltinCPInstruction}:
 * parse-time validation, execution, and lineage handling are owned here rather than as inline opcode branches in the
 * shared class, since the DP release charges a privacy budget and draws fresh randomness on every call - side effects
 * that don't fit the shared class's other (pure, replayable) opcodes.
 *
 * The DP math itself (transform construction, noise generation, sigma calibration) lives in {@link DPBuiltinOps}, which
 * this class calls into.
 */
public class DPBuiltinCPInstruction extends ParameterizedBuiltinCPInstruction {

	public DPBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out, String opcode,
		String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	static DPBuiltinCPInstruction parse(String[] parts, LinkedHashMap<String, String> paramsMap, CPOperand out,
		String opcode, String istr) {
		InstructionUtils.checkNumFields(parts, 5, 6); // laplace=5, gaussian=6
		if(!paramsMap.containsKey("query"))
			throw new DMLRuntimeException(opcode + ": missing 'query'");
		if(!paramsMap.containsKey("sensitivity"))
			throw new DMLRuntimeException(opcode + ": missing 'sensitivity'");
		if(!paramsMap.containsKey("epsilon"))
			throw new DMLRuntimeException(opcode + ": missing 'epsilon'");
		if(opcode.equalsIgnoreCase(Opcodes.DP_GAUSSIAN.toString()) && !paramsMap.containsKey("delta"))
			throw new DMLRuntimeException(opcode + ": missing 'delta'");
		return new DPBuiltinCPInstruction(null, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		String target = params.get("target");
		MatrixBlock X = ec.getMatrixInput(target);
		MatrixBlock outBlock = DPBuiltinOps.release(X, opcode, params, ec.getDPBudgetAccountant());
		ec.releaseMatrixInput(target);
		ec.setMatrixOutput(output.getName(), outBlock);
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		// dp_laplace/dp_gaussian draw fresh randomness and charge a privacy-budget side effect on every
		// call, so a cached lineage-based reuse of a prior release would be unsound.
		throw new DMLRuntimeException(getOpcode() + ": lineage tracing not supported (draws fresh randomness "
			+ "and charges a privacy budget on every call)");
	}
}
