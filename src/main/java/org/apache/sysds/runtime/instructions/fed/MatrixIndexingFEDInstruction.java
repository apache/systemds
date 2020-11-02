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
		FederationMap.FType fedType = fedMapping.getType();

		Map <String, IndexRange> ixs = new HashMap<>();

		for (int i = 0; i < fedMapping.getFederatedRanges().length; i++) {
			long rs = fedMapping.getFederatedRanges()[i].getBeginDims()[0], re = fedMapping.getFederatedRanges()[i]
				.getEndDims()[0], cs = fedMapping.getFederatedRanges()[i].getBeginDims()[1], ce = fedMapping.getFederatedRanges()[i].getEndDims()[1];

			long rsn = 0, ren = 0, csn = 0, cen = 0;

			switch(fedType) {
				case ROW:
					rsn = (ixrange.rowStart >= rs && ixrange.rowStart < re) ? (ixrange.rowStart - rs) : 0;
					ren = (ixrange.rowEnd >= rs && ixrange.rowEnd < re) ? (ixrange.rowEnd - rs) : (re - rs - 1);
					csn = ixrange.colStart;
					cen = ixrange.colEnd;

					if((ixrange.rowStart < re) && (ixrange.rowEnd >= rs)) {
						fedMapping.getFederatedRanges()[i].setBeginDim(0, i != 0 ? fedMapping.getFederatedRanges()[i - 1].getEndDims()[0] : 0);
						fedMapping.getFederatedRanges()[i].setEndDim(0, ren - rsn + 1 + (i == 0 ? 0 : fedMapping.getFederatedRanges()[i - 1].getEndDims()[0]));
						fedMapping.getFederatedRanges()[i].setEndDim(1, cen + 1);
					}
					else {
						fedMapping.getFederatedRanges()[i].setBeginDim(0, i != 0 ? fedMapping.getFederatedRanges()[i - 1].getEndDims()[0] : 0);
						fedMapping.getFederatedRanges()[i].setEndDim(0, fedMapping.getFederatedRanges()[i - 1].getEndDims()[0]);
						fedMapping.getFederatedRanges()[i].setEndDim(1, cen + 1);
						rsn = -1;
						ren = rsn;
						csn = rsn;
						cen = rsn;
					}

					break;
				case COL:
					rsn = ixrange.rowStart;
					ren = ixrange.rowEnd;
					csn = (ixrange.colStart >= cs && ixrange.colStart < ce) ? (ixrange.colStart - cs) : 0;
					cen = (ixrange.colEnd >= cs && ixrange.colEnd < ce) ? (ixrange.colEnd - cs) : (ce - cs - 1);
					if((ixrange.colStart < ce) && (ixrange.colEnd >= cs)) {
						fedMapping.getFederatedRanges()[i].setBeginDim(1, i != 0 ? fedMapping.getFederatedRanges()[i - 1].getEndDims()[1] : 0);
						fedMapping.getFederatedRanges()[i].setEndDim(0, ren + 1);
						fedMapping.getFederatedRanges()[i].setEndDim(1, cen - csn + 1 + (i == 0 ? 0 : fedMapping.getFederatedRanges()[i - 1].getEndDims()[1]));
					}
					else {
						fedMapping.getFederatedRanges()[i].setBeginDim(1, i != 0 ? fedMapping.getFederatedRanges()[i - 1].getEndDims()[1] : 0);
						fedMapping.getFederatedRanges()[i].setEndDim(0, ren + 1);
						fedMapping.getFederatedRanges()[i].setEndDim(1, fedMapping.getFederatedRanges()[i - 1].getEndDims()[1]);
						rsn = -1;
						ren = rsn;
						csn = rsn;
						cen = rsn;
					}
					break;
				case OTHER:
					throw new DMLRuntimeException("Unsupported fed type.");
			}

			ixs.put(fedMapping.getFederatedRanges()[i].toString(), new IndexRange(rsn, ren, csn, cen));
		}

		long varID = FederationUtils.getNextFedDataID();
		FederationMap sortedMapping = fedMapping.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new SliceMatrix(data.getVarID(), varID, ixs.get(range.toString())))).get();
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
		System.out.println(1);
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