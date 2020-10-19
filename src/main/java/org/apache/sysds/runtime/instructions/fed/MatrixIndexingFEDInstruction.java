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

import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.util.IndexRange;

public final class MatrixIndexingFEDInstruction extends IndexingFEDInstruction {
	public MatrixIndexingFEDInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	//for left indexing
	protected MatrixIndexingFEDInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
		CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {

		String opcode = getOpcode();
		IndexRange ixrange = getIndexRange(ec);
		MatrixObject mo = ec.getMatrixObject(input1.getName());

		if (mo.getNumRows()-1 < ixrange.rowEnd || mo.getNumColumns()-1 < ixrange.colEnd ||
			ixrange.rowStart < 0 || ixrange.colStart < 0 ||
			ixrange.rowStart > ixrange.rowEnd || ixrange.colStart > ixrange.colEnd)
			throw new DMLRuntimeException("Federated Matrix Indexing: Invalid indices.");

		FederationMap fedMapping = mo.getFedMapping();
		FederationMap.FType fType = mo.getFedMapping().getType();

		if(! (mo.isFederated() && opcode.equalsIgnoreCase(RightIndex.OPCODE)) ||
			fType == FederationMap.FType.OTHER)
			throw new DMLRuntimeException("Federated Matrix Indexing: "
				+ "Federated input expected, but invoked w/ " + mo.isFederated());

		int dim = fType == FederationMap.FType.ROW ? 0 : 1;
		long ixStart = dim == 0 ? ixrange.rowStart + 1 : ixrange.colStart + 1;
		long ixEnd = dim == 0 ? ixrange.rowEnd + 1 : ixrange.colEnd + 1;

		for(FederatedRange f:fedMapping.getFederatedRanges()) {
			long dimBegin = f.getBeginDims()[dim], dimEnd = f.getEndDims()[dim];
			long from = 0, to = 0;

			if(dimBegin <= ixStart && ixEnd <= dimEnd) {
				from = ixStart; to = ixEnd;
			} //partly starts
			else if(dimEnd >= ixStart && dimBegin <= ixStart && ixEnd >= dimEnd) {
				from = ixStart; to = dimEnd;
			} //partly ends
			else if(dimEnd >= ixEnd && dimBegin <= ixEnd && ixStart <= dimBegin) {
				from = dimBegin; to = ixEnd;
			} //middle
			else if(dimBegin >= ixStart && ixEnd >= dimEnd) {
				from = dimBegin; to = dimEnd;
			}
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.ROW ? 3:5,
				String.valueOf(from).concat(".SCALAR.INT64.true"));
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.ROW ? 4:6,
				String.valueOf(to).concat(".SCALAR.INT64.true"));

			// not really necessary, but just to have scalars instead variables everywhere
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.COL ? 3:5,
				String.valueOf(fType == FederationMap.FType.COL ? ixrange.rowStart + 1 : ixrange.colStart + 1).concat(".SCALAR.INT64.true"));
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.COL ? 3:5,
				String.valueOf(fType == FederationMap.FType.COL ? ixrange.rowEnd + 1 : ixrange.colEnd + 1).concat(".SCALAR.INT64.true"));


			//TODO fed request with new or try with response
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1},
				new long[]{mo.getFedMapping().getID()});
			FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
			FederatedRequest fr3 = mo.getFedMapping().cleanup(getTID(), fr1.getID(), fr2.getID());
			mo.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo.getDataCharacteristics());
			out.setFedMapping(mo.getFedMapping().copyWithNewID(fr1.getID()));

		}
//		out.getFedMapping().setType(fType);
	}

	private void udf(ExecutionContext ec) {
		// acquire locks
		MatrixObject mo = ec.getMatrixObject(input1.getName());
		IndexRange ixrange = getIndexRange(ec);
		FederationMap fedMapping = mo.getFedMapping();

		long varID = FederationUtils.getNextFedDataID();

		FederationMap.FType fType = mo.getFedMapping().getType();

		int dim = fType == FederationMap.FType.ROW ? 0 : 1;
		long ixStart = dim == 0 ? ixrange.rowStart + 1 : ixrange.colStart + 1;
		long ixEnd = dim == 0 ? ixrange.rowEnd + 1 : ixrange.colEnd + 1;

		FederationMap outMapping = fedMapping.mapParallel(varID, (range, data) -> {
			long dimBegin = range.getBeginDims()[dim], dimEnd = range.getEndDims()[dim];
			long from = 0, to = 0;

			if(dimBegin <= ixStart && ixEnd <= dimEnd) {
				from = ixStart; to = ixEnd;
			} //partly starts
			else if(dimEnd >= ixStart && dimBegin <= ixStart && ixEnd >= dimEnd) {
				from = ixStart; to = dimEnd;
			} //partly ends
			else if(dimEnd >= ixEnd && dimBegin <= ixEnd && ixStart <= dimBegin) {
				from = dimBegin; to = ixEnd;
			} //middle
			else if(dimBegin >= ixStart && ixEnd >= dimEnd) {
				from = dimBegin; to = dimEnd;
			}
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.ROW ? 3:5,
				String.valueOf(from).concat(".SCALAR.INT64.true"));
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.ROW ? 4:6,
				String.valueOf(to).concat(".SCALAR.INT64.true"));

			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.COL ? 3:5,
				String.valueOf(fType == FederationMap.FType.COL ? ixrange.rowStart + 1 : ixrange.colStart + 1).concat(".SCALAR.INT64.true"));
			instString = InstructionUtils.replaceOperand(instString, fType == FederationMap.FType.COL ? 3:5,
				String.valueOf(fType == FederationMap.FType.COL ? ixrange.rowEnd + 1 : ixrange.colEnd + 1).concat(".SCALAR.INT64.true"));

			FederatedResponse response;
			try {
				FederatedRequest fr1 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1},
					new long[]{mo.getFedMapping().getID()});
				response = data.executeFederatedOperation(
					new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID())).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		// construct a federated matrix with the encoded data
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo.getDataCharacteristics());
		// set the federated mapping for the matrix
		out.setFedMapping(outMapping);
	}
}
