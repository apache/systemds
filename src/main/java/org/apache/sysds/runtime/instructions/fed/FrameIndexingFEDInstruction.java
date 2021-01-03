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

import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.util.IndexRange;

public final class FrameIndexingFEDInstruction extends IndexingFEDInstruction {
	private static final Log LOG = LogFactory.getLog(FrameIndexingFEDInstruction.class.getName());

	public FrameIndexingFEDInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		rightIndexing(ec);
	}

	private void rightIndexing(ExecutionContext ec)
	{
		//get input and requested index range
		FrameObject in = ec.getFrameObject(input1);
		IndexRange ixrange = getIndexRange(ec);

		//prepare output federation map (copy-on-write)
		FederationMap fedMap = in.getFedMapping().filter(ixrange);

		//modify federated ranges in place
		String[] instStrings = new String[fedMap.getSize()];

		// replace old reshape values for each worker
		int i = 0;
		for(FederatedRange range : fedMap.getMap().keySet()) {
			long rs = range.getBeginDims()[0], re = range.getEndDims()[0],
				cs = range.getBeginDims()[1], ce = range.getEndDims()[1];
			long rsn = (ixrange.rowStart >= rs) ? (ixrange.rowStart - rs) : 0;
			long ren = (ixrange.rowEnd >= rs && ixrange.rowEnd < re) ? (ixrange.rowEnd - rs) : (re - rs - 1);
			long csn = (ixrange.colStart >= cs) ? (ixrange.colStart - cs) : 0;
			long cen = (ixrange.colEnd >= cs && ixrange.colEnd < ce) ? (ixrange.colEnd - cs) : (ce - cs - 1);

			range.setBeginDim(0, Math.max(rs - ixrange.rowStart, 0));
			range.setBeginDim(1, Math.max(cs - ixrange.colStart, 0));
			range.setEndDim(0, (ixrange.rowEnd >= re ? re-ixrange.rowStart : ixrange.rowEnd-ixrange.rowStart + 1));
			range.setEndDim(1, (ixrange.colEnd >= ce ? ce-ixrange.colStart : ixrange.colEnd-ixrange.colStart + 1));

			long[] newIx = new long[]{rsn, ren, csn, cen};

			// change 4 indices
			instStrings[i] = instString;
			String[] instParts = instString.split(Lop.OPERAND_DELIMITOR);
			for(int j = 3; j < 7; j++) {
				instParts[j] = instParts[j]
					.replace(instParts[j].split(Lop.VALUETYPE_PREFIX)[0], String.valueOf(newIx[j-3]+1));
				instStrings[i] = String.join(Lop.OPERAND_DELIMITOR, instParts);
			}
			i++;
		}
		FederatedRequest[] fr1 = FederationUtils.callInstruction(instStrings,
			output, new CPOperand[] {input1}, new long[] {fedMap.getID()});
		fedMap.execute(getTID(), true, fr1, new FederatedRequest[0]);

		//TODO set schema  in for loop if dims are changed
		FrameObject out = ec.getFrameObject(output);
		out.setSchema(in.getSchema());
		out.getDataCharacteristics().setDimension(fedMap.getMaxIndexInRange(0), fedMap.getMaxIndexInRange(1));
		out.setFedMapping(fedMap.copyWithNewID(fr1[0].getID()));
	}
}