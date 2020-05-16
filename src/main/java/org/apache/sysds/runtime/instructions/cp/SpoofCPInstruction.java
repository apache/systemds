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

import java.util.ArrayList;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.lineage.LineageCodegenItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class SpoofCPInstruction extends ComputationCPInstruction {
	private final Class<?> _class;
	private final SpoofOperator _op;
	private final int _numThreads;
	private final CPOperand[] _in;

	private SpoofCPInstruction(SpoofOperator op, Class<?> cla, int k, CPOperand[] in, CPOperand out, String opcode,
			String str) {
		super(CPType.SpoofFused, null, null, null, out, opcode, str);
		_class = cla;
		_op = op;
		_numThreads = k;
		_in = in;
	}

	public Class<?> getOperatorClass() {
		return _class;
	}

	public static SpoofCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		ArrayList<CPOperand> inlist = new ArrayList<>();
		Class<?> cla = CodegenUtils.getClass(parts[1]);
		SpoofOperator op = CodegenUtils.createInstance(cla);
		String opcode =  parts[0] + op.getSpoofType();
		
		for( int i=2; i<parts.length-2; i++ )
			inlist.add(new CPOperand(parts[i]));
		CPOperand out = new CPOperand(parts[parts.length-2]);
		int k = Integer.parseInt(parts[parts.length-1]);
		
		return new SpoofCPInstruction(op, cla, k, inlist.toArray(new CPOperand[0]), out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//get input matrices and scalars, incl pinning of matrices
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		ArrayList<ScalarObject> scalars = new ArrayList<>();
		for (CPOperand input : _in) {
			if(input.getDataType()==DataType.MATRIX)
				inputs.add(ec.getMatrixInput(input.getName()));
			else if(input.getDataType()==DataType.SCALAR) {
				//note: even if literal, it might be compiled as scalar placeholder
				scalars.add(ec.getScalarInput(input));
			}
		}
		
		// set the output dimensions to the hop node matrix dimensions
		if( output.getDataType() == DataType.MATRIX) {
			MatrixBlock out = _op.execute(inputs, scalars, new MatrixBlock(), _numThreads);
			ec.setMatrixOutput(output.getName(), out);
		}
		else if (output.getDataType() == DataType.SCALAR) {
			ScalarObject out = _op.execute(inputs, scalars, _numThreads);
			ec.setScalarOutput(output.getName(), out);
		}
		
		// release input matrices
		for (CPOperand input : _in)
			if(input.getDataType()==DataType.MATRIX)
				ec.releaseMatrixInput(input.getName());
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		//read and deepcopy the corresponding lineage DAG (pre-codegen)
		LineageItem LIroot = LineageCodegenItem.getCodegenLTrace(getOperatorClass().getName()).deepCopy();
		
		//replace the placeholders with original instruction inputs. 
		LineageItemUtils.replaceDagLeaves(ec, LIroot, _in);

		return Pair.of(output.getName(), LIroot);
	}
}
