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

	private void rightIndexing(ExecutionContext ec)
	{
		//get input and requested index range
		MatrixObject in = ec.getMatrixObject(input1);
		IndexRange ixrange = getIndexRange(ec);
		
		//prepare output federation map (copy-on-write)
		FederationMap fedMap = in.getFedMapping().filter(ixrange);
		
		//modify federated ranges in place
		Map<FederatedRange, IndexRange> ixs = new HashMap<>();
		for(FederatedRange range : fedMap.getFedMapping().keySet()) {
			long rs = range.getBeginDims()[0], re = range.getEndDims()[0],
				cs = range.getBeginDims()[1], ce = range.getEndDims()[1];
			long rsn = (ixrange.rowStart >= rs) ? (ixrange.rowStart - rs) : 0;
			long ren = (ixrange.rowEnd >= rs && ixrange.rowEnd < re) ? (ixrange.rowEnd - rs) : (re - rs - 1);
			long csn = (ixrange.colStart >= cs) ? (ixrange.colStart - cs) : 0;
			long cen = (ixrange.colEnd >= cs && ixrange.colEnd < ce) ? (ixrange.colEnd - cs) : (ce - cs - 1);
			if(LOG.isDebugEnabled()) {
				LOG.debug("Ranges for fed location: " + rsn + " " + ren + " " + csn + " " + cen);
				LOG.debug("ixRange                : " + ixrange);
				LOG.debug("Fed Mapping            : " + range);
			}
			range.setBeginDim(0, Math.max(rs - ixrange.rowStart, 0));
			range.setBeginDim(1, Math.max(cs - ixrange.colStart, 0));
			range.setEndDim(0, (ixrange.rowEnd >= re ? re-ixrange.rowStart : ixrange.rowEnd-ixrange.rowStart + 1));
			range.setEndDim(1, (ixrange.colEnd >= ce ? ce-ixrange.colStart : ixrange.colEnd-ixrange.colStart + 1));
			if(LOG.isDebugEnabled())
				LOG.debug("Fed Mapping After      : " + range);
			ixs.put(range, new IndexRange(rsn, ren, csn, cen));
		}

		// execute slicing of valid range 
		long varID = FederationUtils.getNextFedDataID();
		FederationMap slicedFedMap = fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new SliceMatrix(data.getVarID(), varID, ixs.get(range)))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		//update output mapping and data characteristics
		MatrixObject sliced = ec.getMatrixObject(output);
		sliced.getDataCharacteristics()
			.set(slicedFedMap.getMaxIndexInRange(0), slicedFedMap.getMaxIndexInRange(1), (int) in.getBlocksize());
		sliced.setFedMapping(slicedFedMap);
		
		//TODO is this really necessary
		if(ixrange.rowEnd - ixrange.rowStart == 0)
			slicedFedMap.setType(FederationMap.FType.COL);
		else if(ixrange.colEnd - ixrange.colStart == 0)
			slicedFedMap.setType(FederationMap.FType.ROW);
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
			MatrixBlock res = mb.slice(_ixrange, new MatrixBlock());
			MatrixObject mout = ExecutionContext.createMatrixObject(res);
			ec.setVariable(String.valueOf(_outputID), mout);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS_EMPTY);
		}
	}
}
