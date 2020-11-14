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

	private void rightIndexing(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMapping = in.getFedMapping();
		IndexRange ixrange = getIndexRange(ec);
		// FederationMap.FType fedType;
		Map<FederatedRange, IndexRange> ixs = new HashMap<>();

		for(int i = 0; i < fedMapping.getFederatedRanges().length; i++) {
			FederatedRange curFedRange = fedMapping.getFederatedRanges()[i];
			long rs = curFedRange.getBeginDims()[0], re = curFedRange.getEndDims()[0],
				cs = curFedRange.getBeginDims()[1], ce = curFedRange.getEndDims()[1];

			if((ixrange.colStart <= ce) && (ixrange.colEnd >= cs) && (ixrange.rowStart <= re) && (ixrange.rowEnd >= rs)) {
				// If the indexing range contains values that are within the specific federated range.
				// change the range.
				long rsn = (ixrange.rowStart >= rs) ? (ixrange.rowStart - rs) : 0;
				long ren = (ixrange.rowEnd >= rs && ixrange.rowEnd < re) ? (ixrange.rowEnd - rs) : (re - rs - 1);
				long csn = (ixrange.colStart >= cs) ? (ixrange.colStart - cs) : 0;
				long cen = (ixrange.colEnd >= cs && ixrange.colEnd < ce) ? (ixrange.colEnd - cs) : (ce - cs - 1);
				if(LOG.isDebugEnabled()) {
					LOG.debug("Ranges for fed location: " + rsn + " " + ren + " " + csn + " " + cen);
					LOG.debug("ixRange                : " + ixrange);
					LOG.debug("Fed Mapping            : " + curFedRange);
				}
				curFedRange.setBeginDim(0, Math.max(rs - ixrange.rowStart, 0));
				curFedRange.setBeginDim(1, Math.max(cs - ixrange.colStart, 0));
				curFedRange.setEndDim(0,
					(ixrange.rowEnd >= re ? re - ixrange.rowStart : ixrange.rowEnd - ixrange.rowStart + 1));
				curFedRange.setEndDim(1,
					(ixrange.colEnd >= ce ? ce - ixrange.colStart : ixrange.colEnd - ixrange.colStart + 1));
				if(LOG.isDebugEnabled()) {
					LOG.debug("Fed Mapping After      : " + curFedRange);
				}
				ixs.put(curFedRange, new IndexRange(rsn, ren, csn, cen));
			}
			else {
				// If not within the range, change the range to become an 0 times 0 big range.
				// by setting the end dimensions to the same as the beginning dimensions.
				curFedRange.setBeginDim(0, 0);
				curFedRange.setBeginDim(1, 0);
				curFedRange.setEndDim(0, 0);
				curFedRange.setEndDim(1, 0);
			}

		}

		long varID = FederationUtils.getNextFedDataID();
		FederationMap slicedMapping = fedMapping.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new SliceMatrix(data.getVarID(), varID, ixs.getOrDefault(range, new IndexRange(-1, -1, -1, -1)))))
					.get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		MatrixObject sliced = ec.getMatrixObject(output);
		sliced.getDataCharacteristics()
			.set(fedMapping.getMaxIndexInRange(0), fedMapping.getMaxIndexInRange(1), (int) in.getBlocksize());
		if(ixrange.rowEnd - ixrange.rowStart == 0) {
			slicedMapping.setType(FederationMap.FType.COL);
		}
		else if(ixrange.colEnd - ixrange.colStart == 0) {
			slicedMapping.setType(FederationMap.FType.ROW);
		}
		sliced.setFedMapping(slicedMapping);
		LOG.debug(slicedMapping);
		LOG.debug(sliced);
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

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			MatrixBlock res;
			if(_ixrange.rowStart != -1)
				res = mb.slice(_ixrange, new MatrixBlock());
			else
				res = new MatrixBlock();
			MatrixObject mout = ExecutionContext.createMatrixObject(res);
			ec.setVariable(String.valueOf(_outputID), mout);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS_EMPTY);
		}
	}
}
