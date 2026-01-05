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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCIOHandler;
import org.apache.sysds.runtime.ooc.stream.OOCSourceStream;

public class ReblockOOCInstruction extends ComputationOOCInstruction {
	private int blen;

	private ReblockOOCInstruction(Operator op, CPOperand in, CPOperand out, 
		int br, int bc, String opcode, String instr)
	{
		super(OOCType.Reblock, op, in, out, opcode, instr);
		blen = br;
		blen = bc;
	}

	public static ReblockOOCInstruction parseInstruction(String str) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equals(Opcodes.RBLK.toString()))
			throw new DMLRuntimeException("Incorrect opcode for ReblockOOCInstruction:" + opcode);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen=Integer.parseInt(parts[3]);
		return new ReblockOOCInstruction(null, in, out, blen, blen, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//set the output characteristics
		MatrixObject min = ec.getMatrixObject(input1);
		DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), blen, mc.getNonZeros());

		//get the source format from the meta data
		//MetaDataFormat iimd = (MetaDataFormat) min.getMetaData();
		//TODO support other formats than binary 
		
		//create queue, spawn thread for asynchronous reading, and return
		OOCStream<IndexedMatrixValue> q = new OOCSourceStream();
		OOCIOHandler io = OOCCacheManager.getIOHandler();
		OOCIOHandler.SourceReadRequest req = new OOCIOHandler.SourceReadRequest(
			min.getFileName(), Types.FileFormat.BINARY, mc.getRows(), mc.getCols(), blen, mc.getNonZeros(),
			Long.MAX_VALUE, true, q);
		io.scheduleSourceRead(req).whenComplete((res, err) -> {
			if (err != null) {
				Exception ex = err instanceof Exception ? (Exception) err : new Exception(err);
				q.propagateFailure(new DMLRuntimeException(ex));
			}
		});
		
		MatrixObject mout = ec.getMatrixObject(output);
		mout.setStreamHandle(q);
	}
}
