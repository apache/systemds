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

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.IndexRange;

public final class MatrixIndexingFEDInstruction extends IndexingFEDInstruction {
	private static final Log LOG = LogFactory.getLog(MatrixIndexingFEDInstruction.class.getName());

	public MatrixIndexingFEDInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		rightIndexing(ec);
	}


	private void rightIndexing (ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMapping = in.getFedMapping();
		IndexRange ixrange = getIndexRange(ec);
		FederationMap.FType fedType;
		Map <FederatedRange, IndexRange> ixs = new HashMap<>();

		FederatedRange nextDim = new FederatedRange(new long[]{0, 0}, new long[]{0, 0});

		for (int i = 0; i < fedMapping.getFederatedRanges().length; i++) {
			long rs = fedMapping.getFederatedRanges()[i].getBeginDims()[0], re = fedMapping.getFederatedRanges()[i]
				.getEndDims()[0], cs = fedMapping.getFederatedRanges()[i].getBeginDims()[1], ce = fedMapping.getFederatedRanges()[i].getEndDims()[1];

			// for OTHER
			fedType = ((i + 1) < fedMapping.getFederatedRanges().length &&
				fedMapping.getFederatedRanges()[i].getEndDims()[0] == fedMapping.getFederatedRanges()[i+1].getBeginDims()[0]) ?
				FederationMap.FType.ROW : FederationMap.FType.COL;

			long rsn = 0, ren = 0, csn = 0, cen = 0;

			rsn = (ixrange.rowStart >= rs && ixrange.rowStart < re) ? (ixrange.rowStart - rs) : 0;
			ren = (ixrange.rowEnd >= rs && ixrange.rowEnd < re) ? (ixrange.rowEnd - rs) : (re - rs - 1);
			csn = (ixrange.colStart >= cs && ixrange.colStart < ce) ? (ixrange.colStart - cs) : 0;
			cen = (ixrange.colEnd >= cs && ixrange.colEnd < ce) ? (ixrange.colEnd - cs) : (ce - cs - 1);

			fedMapping.getFederatedRanges()[i].setBeginDim(0, i != 0 ? nextDim.getBeginDims()[0] : 0);
			fedMapping.getFederatedRanges()[i].setBeginDim(1, i != 0 ? nextDim.getBeginDims()[1] : 0);
			if((ixrange.colStart < ce) && (ixrange.colEnd >= cs) && (ixrange.rowStart < re) && (ixrange.rowEnd >= rs)) {
				fedMapping.getFederatedRanges()[i].setEndDim(0, ren - rsn + 1 + nextDim.getBeginDims()[0]);
				fedMapping.getFederatedRanges()[i].setEndDim(1,  cen - csn + 1 + nextDim.getBeginDims()[1]);

				ixs.put(fedMapping.getFederatedRanges()[i], new IndexRange(rsn, ren, csn, cen));
			} else {
				fedMapping.getFederatedRanges()[i].setEndDim(0,  i != 0 ? nextDim.getBeginDims()[0] : 0);
				fedMapping.getFederatedRanges()[i].setEndDim(1,  i != 0 ? nextDim.getBeginDims()[1] : 0);
			}

			if(fedType == FederationMap.FType.ROW) {
				nextDim.setBeginDim(0,fedMapping.getFederatedRanges()[i].getEndDims()[0]);
				nextDim.setBeginDim(1, fedMapping.getFederatedRanges()[i].getBeginDims()[1]);
			} else if(fedType == FederationMap.FType.COL) {
				nextDim.setBeginDim(1,fedMapping.getFederatedRanges()[i].getEndDims()[1]);
				nextDim.setBeginDim(0, fedMapping.getFederatedRanges()[i].getBeginDims()[0]);
			}
		}

		long varID = FederationUtils.getNextFedDataID();
		FederationMap sortedMapping = fedMapping.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					-1, new SliceMatrix(data.getVarID(), varID, ixs.getOrDefault(range, new IndexRange(-1, -1, -1, -1))))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});


		MatrixObject sorted = ec.getMatrixObject(output);
		sorted.getDataCharacteristics().set(fedMapping.getMaxIndexInRange(0), fedMapping.getMaxIndexInRange(1), (int) in.getBlocksize());
		sorted.setFedMapping(sortedMapping);

	}


	private static class SliceMatrix extends FederatedUDF {

		private static final long serialVersionUID = 5956832933333848772L;
		private final long _outputID;
		private final IndexRange _ixrange;

		private SliceMatrix(long input, long outputID, IndexRange ixrange) {
			super(new long[] {input});
			_outputID = outputID;
			_ixrange = ixrange;
		}


		@Override public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			MatrixBlock res;
			if(_ixrange.rowStart != -1)
				res = mb.slice(_ixrange, new MatrixBlock());
			else res = new MatrixBlock();
			MatrixObject mout = ExecutionContext.createMatrixObject(res);
			ec.setVariable(String.valueOf(_outputID), mout);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS_EMPTY);
		}
	}
}