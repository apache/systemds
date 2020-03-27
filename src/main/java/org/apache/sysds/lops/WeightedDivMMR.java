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
import org.tugraz.sysds.lops.WeightedDivMM.WDivMMType;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;

public class WeightedDivMMR extends Lop 
{
	public static final String OPCODE = "redwdivmm";
	
	private WDivMMType _weightsType = null;
	private boolean _cacheU = false;
	private boolean _cacheV = false;
	
	public WeightedDivMMR(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, WDivMMType wt, boolean cacheU, boolean cacheV, ExecType et)  {
		super(Lop.Type.WeightedDivMM, dt, vt);
		addInput(input1); //W
		addInput(input2); //U
		addInput(input3); //V
		addInput(input4); //X
		input1.addOutput(this); 
		input2.addOutput(this);
		input3.addOutput(this);
		input4.addOutput(this);	
		
		_weightsType = wt;
		_cacheU = cacheU;
		_cacheV = cacheV;
		setupLopProperties(et);
	}

	@Override
	public String toString() {
		return "Operation = WeightedDivMMR";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			getInputs().get(2).prepInputOperand(input3),
			getInputs().get(3).prepInputOperand(input4),
			prepOutputOperand(output),
			String.valueOf(_weightsType),
			String.valueOf(_cacheU),
			String.valueOf(_cacheV));
	}
}
