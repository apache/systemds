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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SpoofCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateTernarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.CastSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CtableSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MultiReturnParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuaternarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.SpoofSPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.WriteSPInstruction;

public class FEDInstructionUtils {

	public static boolean noFedRuntimeConversion = false;

	/**
	 * Check and replace CP instructions with federated instructions if the instruction match criteria.
	 *
	 * @param inst The instruction to analyze
	 * @param ec   The Execution Context
	 * @return The potentially modified instruction
	 */
	public static Instruction checkAndReplaceCP(Instruction inst, ExecutionContext ec) {
		if(noFedRuntimeConversion)
			return inst;

		FEDInstruction fedinst = null;

		if(inst instanceof AggregateBinaryCPInstruction)
			fedinst = AggregateBinaryFEDInstruction.parseInstruction((AggregateBinaryCPInstruction) inst, ec);
		else if(inst instanceof MMChainCPInstruction)
			fedinst = MMChainFEDInstruction.parseInstruction((MMChainCPInstruction) inst, ec);
		else if(inst instanceof MMTSJCPInstruction)
			fedinst = TsmmFEDInstruction.parseInstruction((MMTSJCPInstruction) inst, ec);
		else if(inst instanceof UnaryCPInstruction)
			fedinst = UnaryFEDInstruction.parseInstruction((UnaryCPInstruction) inst, ec);
		else if(inst instanceof BinaryCPInstruction)
			fedinst = BinaryFEDInstruction.parseInstruction((BinaryCPInstruction) inst, ec);
		else if(inst instanceof ParameterizedBuiltinCPInstruction)
			fedinst = ParameterizedBuiltinFEDInstruction.parseInstruction((ParameterizedBuiltinCPInstruction) inst, ec);
		else if(inst instanceof MultiReturnParameterizedBuiltinCPInstruction)
			fedinst = MultiReturnParameterizedBuiltinFEDInstruction
				.parseInstruction((MultiReturnParameterizedBuiltinCPInstruction) inst, ec);
		else if(inst instanceof TernaryCPInstruction)
			fedinst = TernaryFEDInstruction.parseInstruction((TernaryCPInstruction) inst, ec);
		else if(inst instanceof VariableCPInstruction)
			fedinst = VariableFEDInstruction.parseInstruction((VariableCPInstruction) inst, ec);
		else if(inst instanceof AggregateTernaryCPInstruction)
			fedinst = AggregateTernaryFEDInstruction.parseInstruction((AggregateTernaryCPInstruction) inst, ec);
		else if(inst instanceof QuaternaryCPInstruction)
			fedinst = QuaternaryFEDInstruction.parseInstruction((QuaternaryCPInstruction) inst, ec);
		else if(inst instanceof SpoofCPInstruction)
			fedinst = SpoofFEDInstruction.parseInstruction((SpoofCPInstruction) inst, ec);
		else if(inst instanceof CtableCPInstruction)
			fedinst = CtableFEDInstruction.parseInstruction((CtableCPInstruction) inst, ec);

		// set thread id for federated context management
		if(fedinst != null) {
			fedinst.setTID(ec.getTID());
			return fedinst;
		}
		
		return inst;

	}

	public static Instruction checkAndReplaceSP(Instruction inst, ExecutionContext ec) {
		if(noFedRuntimeConversion)
			return inst;
		FEDInstruction fedinst = null;
		if(inst instanceof CastSPInstruction)
			fedinst = CastFEDInstruction.parseInstruction((CastSPInstruction) inst, ec);
		else if(inst instanceof WriteSPInstruction) {
			WriteSPInstruction instruction = (WriteSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if(data instanceof CacheableData && ((CacheableData<?>) data).isFederated()) {
				// Write spark instruction can not be executed for federated matrix objects (tries to get rdds which do
				// not exist), therefore we replace the instruction with the VariableCPInstruction.
				return VariableCPInstruction.parseInstruction(instruction.getInstructionString());
			}
		}
		else if(inst instanceof QuaternarySPInstruction)
			fedinst = QuaternaryFEDInstruction.parseInstruction((QuaternarySPInstruction) inst, ec);
		else if(inst instanceof SpoofSPInstruction)
			fedinst = SpoofFEDInstruction.parseInstruction((SpoofSPInstruction) inst, ec);
		else if(inst instanceof UnarySPInstruction)
			fedinst = UnaryFEDInstruction.parseInstruction((UnarySPInstruction) inst, ec);
		else if(inst instanceof BinarySPInstruction)
			fedinst = BinaryFEDInstruction.parseInstruction((BinarySPInstruction) inst, ec);
		else if(inst instanceof ParameterizedBuiltinSPInstruction)
			fedinst = ParameterizedBuiltinFEDInstruction.parseInstruction((ParameterizedBuiltinSPInstruction) inst, ec);
		else if(inst instanceof MultiReturnParameterizedBuiltinSPInstruction)
			fedinst = MultiReturnParameterizedBuiltinFEDInstruction
				.parseInstruction((MultiReturnParameterizedBuiltinSPInstruction) inst, ec);
		else if(inst instanceof TernarySPInstruction)
			fedinst = TernaryFEDInstruction.parseInstruction((TernarySPInstruction) inst, ec);
		else if(inst instanceof AggregateTernarySPInstruction)
			fedinst = AggregateTernaryFEDInstruction.parseInstruction((AggregateTernarySPInstruction) inst, ec);
		else if(inst instanceof CtableSPInstruction)
			fedinst = CtableFEDInstruction.parseInstruction((CtableSPInstruction) inst, ec);

		// set thread id for federated context management
		if(fedinst != null) {
			fedinst.setTID(ec.getTID());
			return fedinst;
		}

		return inst;
	}
}
