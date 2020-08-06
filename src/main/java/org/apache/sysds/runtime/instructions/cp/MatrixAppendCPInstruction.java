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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public final class MatrixAppendCPInstruction extends AppendCPInstruction {

	protected MatrixAppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			AppendType type, String opcode, String istr) {
		super(op, in1, in2, out, type, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//get inputs
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		//check input dimensions
		if( _type == AppendType.CBIND && matBlock1.getNumRows() != matBlock2.getNumRows() ) {
			throw new DMLRuntimeException("Append-cbind is not possible for input matrices " + input1.getName() + " and " + input2.getName()
					+ " with different number of rows: "+matBlock1.getNumRows()+" vs "+matBlock2.getNumRows());
		}
		else if( _type == AppendType.RBIND && matBlock1.getNumColumns() != matBlock2.getNumColumns()) {
			throw new DMLRuntimeException("Append-rbind is not possible for input matrices " + input1.getName() + " and " + input2.getName()
					+ " with different number of columns: "+matBlock1.getNumColumns()+" vs "+matBlock2.getNumColumns());
		}
		//execute append operations (append both inputs to initially empty output)
		MatrixBlock ret = matBlock1.append(matBlock2, new MatrixBlock(), _type==AppendType.CBIND);
		//set output and release inputs 
		ec.setMatrixOutput(output.getName(), ret);
		ec.releaseMatrixInput(input1.getName(), input2.getName());
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		//TODO: break append to cbind and rbind for full compilation chain
		String opcode = _type.toString().toLowerCase();
		return Pair.of(output.getName(),
			new LineageItem(opcode, LineageItemUtils.getLineage(ec, input1, input2)));
	}
}
