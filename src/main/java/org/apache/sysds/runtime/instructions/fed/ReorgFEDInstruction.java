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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;
import java.util.stream.Stream;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.DiagIndex;
import org.apache.sysds.runtime.functionobjects.RevIndex;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReorgSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class ReorgFEDInstruction extends UnaryFEDInstruction {
	// roll-specific attributes
	private CPOperand _shift = null;

	public ReorgFEDInstruction(Operator op, CPOperand in1, CPOperand out, String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.Reorg, op, in1, out, opcode, istr, fedOut);
	}

	public ReorgFEDInstruction(Operator op, CPOperand in1, CPOperand out, String opcode, String istr) {
		super(FEDType.Reorg, op, in1, out, opcode, istr);
	}

	private ReorgFEDInstruction(Operator op, CPOperand in, CPOperand shift, CPOperand out,  String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.Reorg, op, in, shift, out, opcode, istr, fedOut);
		_shift = shift;
	}

	public static ReorgFEDInstruction parseInstruction(ReorgCPInstruction rinst) {
		if (rinst.input2 != null) {
			return new ReorgFEDInstruction(rinst.getOperator(), rinst.input1, rinst.input2, rinst.output, rinst.getOpcode(),
					rinst.getInstructionString(), FederatedOutput.NONE);
		} else{
			return new ReorgFEDInstruction(rinst.getOperator(), rinst.input1, rinst.output, rinst.getOpcode(),
					rinst.getInstructionString(), FederatedOutput.NONE);
		}
	}

	public static ReorgFEDInstruction parseInstruction(ReorgSPInstruction rinst) {
		if (rinst.input2 != null) {
			return new ReorgFEDInstruction(rinst.getOperator(), rinst.input1, rinst.input2, rinst.output, rinst.getOpcode(),
					rinst.getInstructionString(), FederatedOutput.NONE);
		} else{
			return new ReorgFEDInstruction(rinst.getOperator(), rinst.input1, rinst.output, rinst.getOpcode(),
					rinst.getInstructionString(), FederatedOutput.NONE);
		}
	}

	public static ReorgFEDInstruction parseInstruction(String str) {
		CPOperand in = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand out = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		FederatedOutput fedOut;
		if(opcode.equalsIgnoreCase(Opcodes.TRANSPOSE.toString())) {
			InstructionUtils.checkNumFields(str, 2, 3, 4);
			in.split(parts[1]);
			out.split(parts[2]);
			int k = str.startsWith(Types.ExecMode.SPARK.name()) ? 0 : Integer.parseInt(parts[3]);
			fedOut = str.startsWith(Types.ExecMode.SPARK.name()) ? FederatedOutput.valueOf(parts[3]) : FederatedOutput
				.valueOf(parts[4]);
			return new ReorgFEDInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k), in, out, opcode, str,
				fedOut);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.DIAG.toString())) {
			parseUnaryInstruction(str, in, out); // max 2 operands
			fedOut = parseFedOutFlag(str, 3);
			return new ReorgFEDInstruction(new ReorgOperator(DiagIndex.getDiagIndexFnObject()), in, out, opcode, str,
				fedOut);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.REV.toString())) {
			parseUnaryInstruction(str, in, out); // max 2 operands
			fedOut = parseFedOutFlag(str, 3);
			return new ReorgFEDInstruction(new ReorgOperator(RevIndex.getRevIndexFnObject()), in, out, opcode, str,
				fedOut);
		}
		else if (opcode.equalsIgnoreCase(Opcodes.ROLL.toString())) {
			InstructionUtils.checkNumFields(str, 3);
			in.split(parts[1]);
			out.split(parts[3]);
			CPOperand shift = new CPOperand(parts[2]);
			fedOut = parseFedOutFlag(str, 3);
			return new ReorgFEDInstruction(new ReorgOperator(new RollIndex(0)),
					in, out, shift, opcode, str, fedOut);
		}
		else {
			throw new DMLRuntimeException("ReorgFEDInstruction: unsupported opcode: " + opcode);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		ReorgOperator r_op = (ReorgOperator) _optr;
		boolean isSpark = instString.startsWith("SPARK");

		if( !mo1.isFederated() )
			throw new DMLRuntimeException("Federated Reorg: "
				+ "Federated input expected, but invoked w/ "+mo1.isFederated());
		if ( !( mo1.isFederated(FType.COL) || mo1.isFederated(FType.ROW) ) )
			throw new DMLRuntimeException("Federation type " + mo1.getFedMapping().getType()
				+ " is not supported for Reorg processing");

		if(instOpcode.equals(Opcodes.TRANSPOSE.toString())) {
			//execute transpose at federated site
			long id = FederationUtils.getNextFedDataID();
			FederatedRequest fr = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), mo1.getDataType());

			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output, id, new CPOperand[] {input1},
				new long[] {mo1.getFedMapping().getID()}, isSpark ? Types.ExecType.SPARK : Types.ExecType.CP, true);
			Future<FederatedResponse>[] ffr = mo1.getFedMapping().execute(getTID(), true, fr, fr1);

			if (_fedOut != null && !_fedOut.isForcedLocal()){
				//drive output federated mapping
				MatrixObject out = ec.getMatrixObject(output);
				long nnz = (mo1.getNnz() != -1) ? mo1.getNnz() : FederationUtils.sumNonZeros(ffr);
				out.getDataCharacteristics().setDimension(mo1.getNumColumns(), mo1.getNumRows())
					.setBlocksize(mo1.getBlocksize()).setNonZeros(nnz);
				out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1.getID()).transpose());
			} else {
				FederatedRequest getRequest = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
				Future<FederatedResponse>[] execResponse = mo1.getFedMapping().execute(getTID(), true, fr1, getRequest);
				ec.setMatrixOutput(output.getName(),
					FederationUtils.bind(execResponse, mo1.isFederated(FType.ROW)));
			}
		} else if ( mo1.isFederated(FType.PART) ){
			throw new DMLRuntimeException("Operation with opcode " + instOpcode + " is not supported with PART input");
		}
		else if(instOpcode.equalsIgnoreCase(Opcodes.REV.toString())) {
			long id = FederationUtils.getNextFedDataID();
			FederatedRequest fr = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), mo1.getDataType());

			//execute transpose at federated site
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output, id, new CPOperand[] {input1},
				new long[] {mo1.getFedMapping().getID()}, isSpark ? Types.ExecType.SPARK : Types.ExecType.CP, true);
			Future<FederatedResponse>[] ffr = mo1.getFedMapping().execute(getTID(), true, fr, fr1);

			if(mo1.isFederated(FType.ROW))
				mo1.getFedMapping().reverseFedMap();

			//derive output federated mapping
			MatrixObject out = ec.getMatrixObject(output);
			long nnz = (mo1.getNnz() != -1) ? mo1.getNnz() : FederationUtils.sumNonZeros(ffr);
			out.getDataCharacteristics().setDimension(mo1.getNumRows(), mo1.getNumColumns())
				.setBlocksize(mo1.getBlocksize()).setNonZeros(nnz);
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1.getID()));

			optionalForceLocal(out);
		} else if (instOpcode.equalsIgnoreCase(Opcodes.ROLL.toString())) {
			long rlen = mo1.getNumRows();
			long shift = ec.getScalarInput(_shift).getLongValue();
			shift %= (rlen != 0 ? rlen : 1); // roll matrix with axis=none

			long inID = mo1.getFedMapping().getID();
			long outEndID = FederationUtils.getNextFedDataID();
			long outStartID = FederationUtils.getNextFedDataID();

			List<Pair<FederatedRange, FederatedData>> inMap = mo1.getFedMapping().getMap();
			Pair<FederationMap, Long> rollResult = rollFedMap(
				inMap, inID, outEndID, outStartID, shift, rlen, mo1.getFedMapping().getType());
			long length = rollResult.getValue();
			FederationMap outFedMap = rollResult.getKey();

			FederatedRequest frEnd = new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, outEndID,
					new ReorgFEDInstruction.SliceMatrix(inID, outEndID, length, true));
			FederatedRequest frStart = new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, outStartID,
					new ReorgFEDInstruction.SliceMatrix(inID, outStartID, length, false));
			Future<FederatedResponse>[] ffr = outFedMap.executeRoll(getTID(), true, frEnd, frStart, rlen);

			//derive output federated mapping
			MatrixObject out = ec.getMatrixObject(output);
			long nnz = (mo1.getNnz() != -1) ? mo1.getNnz() : FederationUtils.sumNonZeros(ffr);
			out.getDataCharacteristics()
				.setDimension(mo1.getNumRows(), mo1.getNumColumns())
				.setBlocksize(mo1.getBlocksize())
				.setNonZeros(nnz);
			out.setFedMapping(outFedMap);
			optionalForceLocal(out);
		}
		else if (instOpcode.equals(Opcodes.DIAG.toString())) {
			RdiagResult result;
			// diag(diag(X))
			if (mo1.getNumColumns() == 1 && mo1.getNumRows() != 1) {
				result = rdiagV2M(mo1, r_op);
			} else {
				result = rdiagM2V(mo1, r_op);
			}

			FederationMap diagFedMap = updateFedRanges(result);

			//update output mapping and data characteristics
			MatrixObject rdiag = ec.getMatrixObject(output);
			rdiag.getDataCharacteristics()
				.set(diagFedMap.getMaxIndexInRange(0), diagFedMap.getMaxIndexInRange(1), mo1.getBlocksize());
			rdiag.setFedMapping(diagFedMap);
			optionalForceLocal(rdiag);
		}
	}


	public Pair<FederationMap, Long> rollFedMap(List<Pair<FederatedRange, FederatedData>> oldMap, long inID,
												long outEndID, long outStartID, long shift, long rlen, FType type) {
		List<Pair<FederatedRange, FederatedData>> map = new ArrayList<>();
		long length = 0;

		for(Map.Entry<FederatedRange, FederatedData> e : oldMap) {
			if(e.getKey().getSize() == 0) continue;
			FederatedRange fedRange = new FederatedRange(e.getKey());
			long beginRow = fedRange.getBeginDims()[0] + shift;
			long endRow = fedRange.getEndDims()[0] + shift;

			beginRow = beginRow > rlen ? beginRow - rlen : beginRow;
			endRow = endRow > rlen ? endRow - rlen : endRow;

			if (beginRow < endRow) {
				fedRange.setBeginDim(0, beginRow);
				fedRange.setEndDim(0, endRow);
				map.add(Pair.of(fedRange, e.getValue().copyWithNewID(inID)));
			} else {
				length = rlen - beginRow;
				fedRange.setBeginDim(0, beginRow);
				fedRange.setEndDim(0, rlen);
				map.add(Pair.of(fedRange, e.getValue().copyWithNewID(outEndID)));

				FederatedRange startRange = new FederatedRange(fedRange);
				startRange.setBeginDim(0, 0);
				startRange.setEndDim(0, endRow);
				map.add(Pair.of(startRange, e.getValue().copyWithNewID(outStartID)));
			}
		}
		return Pair.of(new FederationMap(outEndID, map, type), length);
	}

	/**
	 * Update the federated ranges of result and return the updated federation map.
	 * @param result RdiagResult for which the fedmap is updated
	 * @return updated federation map
	 */
	private FederationMap updateFedRanges(RdiagResult result){
		FederationMap diagFedMap = result.getFedMap();
		Map<FederatedRange, int[]> dcs = result.getDcs();

		for(int i = 0; i < diagFedMap.getFederatedRanges().length; i++) {
			int[] newRange = dcs.get(diagFedMap.getFederatedRanges()[i]);

			diagFedMap.getFederatedRanges()[i].setBeginDim(0,
				(diagFedMap.getFederatedRanges()[i].getBeginDims()[0] == 0 ||
					i == 0) ? 0 : diagFedMap.getFederatedRanges()[i - 1].getEndDims()[0]);
			diagFedMap.getFederatedRanges()[i].setEndDim(0,
				diagFedMap.getFederatedRanges()[i].getBeginDims()[0] + newRange[0]);
			diagFedMap.getFederatedRanges()[i].setBeginDim(1,
				(diagFedMap.getFederatedRanges()[i].getBeginDims()[1] == 0 ||
					i == 0) ? 0 : diagFedMap.getFederatedRanges()[i - 1].getEndDims()[1]);
			diagFedMap.getFederatedRanges()[i].setEndDim(1,
				diagFedMap.getFederatedRanges()[i].getBeginDims()[1] + newRange[1]);
		}
		return diagFedMap;
	}

	/**
	 * If federated output is forced local, the output will be retrieved and removed from federated workers.
	 * @param outputMatrixObject which will be retrieved and removed from federated workers
	 */
	private void optionalForceLocal(MatrixObject outputMatrixObject){
		if ( _fedOut != null && _fedOut.isForcedLocal() ){
			outputMatrixObject.acquireReadAndRelease();
			outputMatrixObject.getFedMapping().cleanup(getTID(), outputMatrixObject.getFedMapping().getID());
		}
	}

	private class RdiagResult {
		FederationMap fedMap;
		Map<FederatedRange, int[]> dcs;

		public RdiagResult(FederationMap fedMap, Map<FederatedRange, int[]> dcs) {
			this.fedMap = fedMap;
			this.dcs = dcs;
		}

		public FederationMap getFedMap() {
			return fedMap;
		}

		public Map<FederatedRange, int[]> getDcs() {
			return dcs;
		}
	}

	private RdiagResult rdiagV2M (MatrixObject mo1, ReorgOperator r_op) {
		FederationMap fedMap = mo1.getFedMapping();
		boolean rowFed = mo1.isFederated(FType.ROW);

		long varID = FederationUtils.getNextFedDataID();
		Map<FederatedRange, int[]> dcs = new HashMap<>();
		FederationMap diagFedMap;

		diagFedMap = fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new ReorgFEDInstruction.DiagMatrix(data.getVarID(),
						varID, r_op,
						rowFed ? (new int[] {range.getBeginDimsInt()[0], range.getEndDimsInt()[0]}) :
							new int[] {range.getBeginDimsInt()[1], range.getEndDimsInt()[1]},
						rowFed, (int) mo1.getNumRows()))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				int[] subRangeCharacteristics = (int[]) response.getData()[0];
				synchronized(dcs) {
					dcs.put(range, subRangeCharacteristics);
				}
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});
		return new RdiagResult(diagFedMap, dcs);
	}

	private RdiagResult rdiagM2V (MatrixObject mo1, ReorgOperator r_op) {
		FederationMap fedMap = mo1.getFedMapping();
		boolean rowFed = mo1.isFederated(FType.ROW);

		long varID = FederationUtils.getNextFedDataID();
		Map<FederatedRange, int[]> dcs = new HashMap<>();
		FederationMap diagFedMap;

		diagFedMap = fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new ReorgFEDInstruction.Rdiag(data.getVarID(), varID, r_op,
						rowFed ? (new int[] {range.getBeginDimsInt()[0], range.getEndDimsInt()[0]}) :
							new int[] {range.getBeginDimsInt()[1], range.getEndDimsInt()[1]},
						rowFed))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				int[] subRangeCharacteristics = (int[]) response.getData()[0];
				synchronized(dcs) {
					dcs.put(range, subRangeCharacteristics);
				}
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});
		return new RdiagResult(diagFedMap, dcs);
	}

	public static class SliceMatrix extends FederatedUDF {
		private static final long serialVersionUID = -3466926635958851402L;
		private final long _outputID;
		private final int _sliceRow;
		private final boolean _isRight;

		private SliceMatrix(long input, long outputID, long sliceRow, boolean isRight) {
			super(new long[] {input});
			_outputID = outputID;
			_sliceRow = (int) sliceRow;
			_isRight = isRight;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock oriBlock = ((MatrixObject) data[0]).acquireReadAndRelease();
			MatrixBlock resBlock;

			if (_sliceRow != 0){
				if (_isRight){
					resBlock = oriBlock.slice(0, _sliceRow-1, 0,
							oriBlock.getNumColumns()-1, new MatrixBlock());
				} else{
					resBlock = oriBlock.slice(_sliceRow, oriBlock.getNumRows()-1,
							0, oriBlock.getNumColumns()-1, new MatrixBlock());
				}
			} else{
				resBlock = oriBlock;
			}
			ec.setMatrixOutput(String.valueOf(_outputID), resBlock);
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}

		@Override
		public List<Long> getOutputIds() {
			return new ArrayList<>(Arrays.asList(_outputID));
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return Pair.of(String.valueOf(_outputID),
					new LineageItem());
		}
	}

	public static class Rdiag extends FederatedUDF {

		private static final long serialVersionUID = -3466926635958851402L;
		private final long _outputID;
		private final ReorgOperator _r_op;
		private final int[] _slice;
		private final boolean _rowFed;

		private Rdiag(long input, long outputID, ReorgOperator r_op, int[] slice, boolean rowFed) {
			super(new long[] {input});
			_outputID = outputID;
			_r_op = r_op;
			_slice = slice;
			_rowFed = rowFed;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			MatrixBlock soresBlock;
			MatrixBlock res;

			soresBlock = _rowFed ?
				mb.slice(0, mb.getNumRows() - 1, _slice[0], _slice[1] - 1, new MatrixBlock()) :
				mb.slice(_slice[0], _slice[1] - 1);
			res = soresBlock.reorgOperations(_r_op, new MatrixBlock(), 0, 0, 0);

			MatrixObject mout = ExecutionContext.createMatrixObject(res);
			mout.setDiag(true);
			ec.setVariable(String.valueOf(_outputID), mout);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new int[]{res.getNumRows(), res.getNumColumns()});
		}

		@Override
		public List<Long> getOutputIds() {
			return new ArrayList<>(Arrays.asList(_outputID));
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			LineageItem[] liUdfInputs = Arrays.stream(getInputIDs())
					.mapToObj(id -> ec.getLineage().get(String.valueOf(id))).toArray(LineageItem[]::new);
			CPOperand r_op = new CPOperand(_r_op.fn.getClass().getSimpleName(), ValueType.STRING, DataType.SCALAR, true);
			CPOperand slice = new CPOperand(Arrays.toString(_slice), ValueType.STRING, DataType.SCALAR, true);
			CPOperand rowFed = new CPOperand(String.valueOf(_rowFed), ValueType.BOOLEAN, DataType.SCALAR, true);
			LineageItem[] otherInputs = LineageItemUtils.getLineage(ec, r_op, slice, rowFed);
			LineageItem[] liInputs = Stream.concat(Arrays.stream(liUdfInputs), Arrays.stream(otherInputs))
					.toArray(LineageItem[]::new);
			return Pair.of(String.valueOf(_outputID), 
					new LineageItem(getClass().getSimpleName(), liInputs));
		}
	}

	public static class DiagMatrix extends FederatedUDF {

		private static final long serialVersionUID = -3466926635958851402L;
		private final long _outputID;
		private final ReorgOperator _r_op;
		private final int _len;
		private final int[] _slice;
		private final boolean _rowFed;

		private DiagMatrix(long input, long outputID, ReorgOperator r_op, int[] slice, boolean rowFed, int len) {
			super(new long[] {input});
			_outputID = outputID;
			_r_op = r_op;
			_len = len;
			_rowFed = rowFed;
			_slice = slice;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			MatrixBlock res;

			MatrixBlock tmp = mb.reorgOperations(_r_op, new MatrixBlock(), 0, 0, 0);
			if(_rowFed) {
				res = new MatrixBlock(mb.getNumRows(), _len, 0.0);
				res.copy(0, res.getNumRows()-1, _slice[0], _slice[1]-1, tmp, false);
			} else {
				res = new MatrixBlock(_len, _slice[1], 0.0);
				res.copy(_slice[0], _slice[1]-1, 0, mb.getNumColumns() - 1, tmp, false);
			}
			MatrixObject mout = ExecutionContext.createMatrixObject(res);
			mout.setDiag(true);
			ec.setVariable(String.valueOf(_outputID), mout);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new int[]{res.getNumRows(), res.getNumColumns()});
		}

		@Override
		public List<Long> getOutputIds() {
			return new ArrayList<>(Arrays.asList(_outputID));
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			LineageItem[] liUdfInputs = Arrays.stream(getInputIDs())
					.mapToObj(id -> ec.getLineage().get(String.valueOf(id))).toArray(LineageItem[]::new);
			CPOperand r_op = new CPOperand(_r_op.fn.getClass().getSimpleName(), ValueType.STRING, DataType.SCALAR, true);
			CPOperand len = new CPOperand(String.valueOf(_len), ValueType.INT32, DataType.SCALAR, true);
			CPOperand slice = new CPOperand(Arrays.toString(_slice), ValueType.STRING, DataType.SCALAR, true);
			CPOperand rowFed = new CPOperand(String.valueOf(_rowFed), ValueType.BOOLEAN, DataType.SCALAR, true);
			LineageItem[] otherInputs = LineageItemUtils.getLineage(ec, r_op, len, slice, rowFed);
			LineageItem[] liInputs = Stream.concat(Arrays.stream(liUdfInputs), Arrays.stream(otherInputs))
					.toArray(LineageItem[]::new);
			return Pair.of(String.valueOf(_outputID), 
					new LineageItem(getClass().getSimpleName(), liInputs));
		}
	}
}
