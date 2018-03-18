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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class MatrixBuiltinNaryCPInstruction extends BuiltinNaryCPInstruction {

	protected MatrixBuiltinNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand[] inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		//pin input matrix blocks
		MatrixBlock in1 = ec.getMatrixInput(inputs[0].getName());
		MatrixBlock[] in2 = new MatrixBlock[inputs.length-1];
		for( int i=1; i<inputs.length; i++ )
			in2[i-1] = ec.getMatrixInput(inputs[i].getName());
		
		MatrixBlock outBlock = null;
		if( "cbind".equals(getOpcode()) || "rbind".equals(getOpcode()) ) {
			boolean cbind = "cbind".equals(getOpcode());
			outBlock = in1.append(in2, new MatrixBlock(), cbind);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode: "+getOpcode());
		}
		
		//release inputs and set output
		for( int i=0; i<inputs.length; i++ )
			ec.releaseMatrixInput(inputs[i].getName());
		ec.setMatrixOutput(output.getName(), outBlock);
	}
}
