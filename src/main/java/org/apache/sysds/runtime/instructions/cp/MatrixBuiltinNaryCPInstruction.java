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

import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.lineage.LineageTraceable;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class MatrixBuiltinNaryCPInstruction extends BuiltinNaryCPInstruction implements LineageTraceable {

	protected MatrixBuiltinNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand[] inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//separate scalars and matrices and pin all input matrices
		List<MatrixBlock> matrices = ec.getMatrixInputs(inputs, true);
		List<ScalarObject> scalars = ec.getScalarInputs(inputs);
		List<FrameBlock> frames = ec.getFrameInputs(inputs);
		
		CacheBlock<?> outBlock = null;
		if( Opcodes.CBIND.toString().equals(getOpcode()) || Opcodes.RBIND.toString().equals(getOpcode()) ) {
			boolean cbind = Opcodes.CBIND.toString().equals(getOpcode());
			if(frames.size() == 0 ) { //matrix/scalar
	 			//robustness for empty lists: create 0-by-0 matrix block
				outBlock = matrices.size() == 0 ? new MatrixBlock(0, 0, 0) : 
					matrices.get(0).append(matrices.subList(1, matrices.size())
						.toArray(new MatrixBlock[0]), new MatrixBlock(), cbind);
			}
			else {
				//TODO native nary frame append
				outBlock = frames.get(0);
				for(int i=1; i<frames.size(); i++)
					outBlock = ((FrameBlock)outBlock).append(frames.get(i), cbind);
			}
		}
		else if( ArrayUtils.contains(new String[]{Opcodes.NMIN.toString(), Opcodes.NMAX.toString(), Opcodes.NP.toString(), Opcodes.NM.toString()}, getOpcode()) ) {
			outBlock = MatrixBlock.naryOperations(_optr, matrices.toArray(new MatrixBlock[0]),
				scalars.toArray(new ScalarObject[0]), new MatrixBlock());
		}
		else {
			throw new DMLRuntimeException("Unknown opcode: "+getOpcode());
		}
		
		//release inputs and set output matrix or scalar
		ec.releaseMatrixInputs(inputs, true);
		ec.releaseFrameInputs(inputs);
		if( output.getDataType().isMatrix()) {
			ec.setMatrixOutput(output.getName(), (MatrixBlock)outBlock);
		}
		else if( output.getDataType().isFrame()) {
			ec.setFrameOutput(output.getName(), (FrameBlock)outBlock);
		}
		else {
			ec.setVariable(output.getName(), ScalarObjectFactory.createScalarObject(
				output.getValueType(), ((MatrixBlock)outBlock).get(0, 0)));
		}
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(),
			new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, inputs)));
	}
}
