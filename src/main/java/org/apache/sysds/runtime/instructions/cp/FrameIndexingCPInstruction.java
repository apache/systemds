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

import org.apache.sysds.common.Opcodes;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.util.IndexRange;

public final class FrameIndexingCPInstruction extends IndexingCPInstruction {

	protected FrameIndexingCPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	protected FrameIndexingCPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		IndexRange ixrange = getIndexRange(ec);
		
		//right indexing
		if( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) ) {

			//execute right indexing operation
			FrameBlock in = ec.getFrameInput(input1.getName());
			FrameBlock out = in.slice(ixrange, new FrameBlock());
			
			//unpin rhs input
			ec.releaseFrameInput(input1.getName());
			
			//unpin output
			ec.setFrameOutput(output.getName(), out);
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString())) {
			FrameBlock lin = ec.getFrameInput(input1.getName());
			FrameBlock out = null;
			
			if(input2.getDataType() == DataType.FRAME) { //FRAME<-FRAME
				FrameBlock rin = ec.getFrameInput(input2.getName());
				out = lin.leftIndexingOperations(rin, ixrange, new FrameBlock());
				ec.releaseFrameInput(input2.getName());
			}
			else { //FRAME<-SCALAR 
				if(!ixrange.isScalar())
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: "+ixrange.toString()+"." );
				ScalarObject scalar = ec.getScalarInput(input2);
				out = new FrameBlock(lin);
				out.set((int)ixrange.rowStart, (int)ixrange.colStart, scalar.getStringValue());
			}

			//unpin lhs input
			ec.releaseFrameInput(input1.getName());
			
			//unpin output
			ec.setFrameOutput(output.getName(), out);
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in FrameIndexingCPInstruction.");		
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, input1,input2,input3,rowLower,rowUpper,colLower,colUpper)));
	}
}
