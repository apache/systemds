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

package org.apache.sysds.runtime.controlprogram.federated;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

public class FederationUtils {
	protected static Logger log = Logger.getLogger(FederationUtils.class);
	private static final IDSequence _idSeq = new IDSequence();

	public static void resetFedDataID() {
		_idSeq.reset();
	}

	public static long getNextFedDataID() {
		return _idSeq.getNextID();
	}

	public static FederatedRequest callInstruction(String inst, CPOperand varOldOut, CPOperand[] varOldIn, long[] varNewIn) {
		//TODO better and safe replacement of operand names --> instruction utils
		long id = getNextFedDataID();
		String linst = inst.replace(ExecType.SPARK.name(), ExecType.CP.name());
		linst = linst.replace(
			Lop.OPERAND_DELIMITOR+varOldOut.getName()+Lop.DATATYPE_PREFIX,
			Lop.OPERAND_DELIMITOR+String.valueOf(id)+Lop.DATATYPE_PREFIX);
		for(int i=0; i<varOldIn.length; i++)
			if( varOldIn[i] != null ) {
				linst = linst.replace(
					Lop.OPERAND_DELIMITOR+varOldIn[i].getName()+Lop.DATATYPE_PREFIX,
					Lop.OPERAND_DELIMITOR+String.valueOf(varNewIn[i])+Lop.DATATYPE_PREFIX);
				linst = linst.replace("="+varOldIn[i].getName(), "="+String.valueOf(varNewIn[i])); //parameterized
			}
		return new FederatedRequest(RequestType.EXEC_INST, id, linst);
	}

	public static MatrixBlock aggAdd(Future<FederatedResponse>[] ffr) {
		try {
			SimpleOperator op = new SimpleOperator(Plus.getPlusFnObject());
			MatrixBlock[] in = new MatrixBlock[ffr.length];
			for(int i=0; i<ffr.length; i++)
				in[i] = (MatrixBlock) ffr[i].get().getData()[0];
			return MatrixBlock.naryOperations(op, in, new ScalarObject[0], new MatrixBlock());
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static MatrixBlock aggMean(Future<FederatedResponse>[] ffr, FederationMap map) {
		try {
			FederatedRange[] ranges = map.getFederatedRanges();
			BinaryOperator bop = InstructionUtils.parseBinaryOperator("+");
			ScalarOperator sop1 = InstructionUtils.parseScalarBinaryOperator("*", false);
			MatrixBlock ret = null;
			long size = 0;
			for(int i=0; i<ffr.length; i++) {
				MatrixBlock tmp = (MatrixBlock)ffr[i].get().getData()[0];
				size += ranges[i].getSize(0);
				sop1 = sop1.setConstant(ranges[i].getSize(0));
				tmp = tmp.scalarOperations(sop1, new MatrixBlock());
				ret = (ret==null) ? tmp : ret.binaryOperationsInPlace(bop, tmp);
			}
			ScalarOperator sop2 = InstructionUtils.parseScalarBinaryOperator("/", false);
			sop2 = sop2.setConstant(size);
			return ret.scalarOperations(sop2, new MatrixBlock());
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static MatrixBlock[] getResults(Future<FederatedResponse>[] ffr) {
		try {
			MatrixBlock[] ret = new MatrixBlock[ffr.length];
			for(int i=0; i<ffr.length; i++)
				ret[i] = (MatrixBlock) ffr[i].get().getData()[0];
			return ret;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static MatrixBlock bind(Future<FederatedResponse>[] ffr, boolean cbind) {
		// TODO handle non-contiguous cases
		try {
			MatrixBlock[] tmp = getResults(ffr);
			return tmp[0].append(
				Arrays.copyOfRange(tmp, 1, tmp.length),
				new MatrixBlock(), cbind);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static MatrixBlock aggMinMax(Future<FederatedResponse>[] ffr, boolean isMin, boolean isScalar, Optional<FederationMap.FType> fedType) {
		try {
			if (!fedType.isPresent() || fedType.get() == FederationMap.FType.OTHER) {
				double res = isMin ? Double.MAX_VALUE : -Double.MAX_VALUE;
				for (Future<FederatedResponse> fr : ffr) {
					double v = isScalar ? ((ScalarObject) fr.get().getData()[0]).getDoubleValue() :
						isMin ? ((MatrixBlock) fr.get().getData()[0]).min() : ((MatrixBlock) fr.get().getData()[0]).max();
					res = isMin ? Math.min(res, v) : Math.max(res, v);
				}
				return new MatrixBlock(1, 1, res);
			} else {
				MatrixBlock[] tmp = getResults(ffr);
				int dim = fedType.get() == FederationMap.FType.COL ? tmp[0].getNumRows() : tmp[0].getNumColumns();

				for (int i = 0; i < ffr.length - 1; i++)
					for (int j = 0; j < dim; j++)
						if (fedType.get() == FederationMap.FType.COL)
							tmp[i + 1].setValue(j, 0, isMin ? Double.min(tmp[i].getValue(j, 0), tmp[i + 1].getValue(j, 0)) :
								Double.max(tmp[i].getValue(j, 0), tmp[i + 1].getValue(j, 0)));
						else tmp[i + 1].setValue(0, j, isMin ? Double.min(tmp[i].getValue(0, j), tmp[i + 1].getValue(0, j)) :
							Double.max(tmp[i].getValue(0, j), tmp[i + 1].getValue(0, j)));
				return tmp[ffr.length-1];
			}
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static ScalarObject aggScalar(AggregateUnaryOperator aop, Future<FederatedResponse>[] ffr) {
		if(!(aop.aggOp.increOp.fn instanceof KahanFunction || (aop.aggOp.increOp.fn instanceof Builtin &&
			(((Builtin) aop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN ||
				((Builtin) aop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX)))) {
			throw new DMLRuntimeException("Unsupported aggregation operator: "
				+ aop.aggOp.increOp.getClass().getSimpleName());
		}

		try {
			if(aop.aggOp.increOp.fn instanceof Builtin){
				// then we know it is a Min or Max based on the previous check.
				boolean isMin = ((Builtin) aop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN;
				return new DoubleObject(aggMinMax(ffr, isMin, true,  Optional.empty()).getValue(0,0));
			}
			else {
				double sum = 0; //uak+
				for( Future<FederatedResponse> fr : ffr )
					sum += ((ScalarObject)fr.get().getData()[0]).getDoubleValue();
				return new DoubleObject(sum);
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static MatrixBlock aggMatrix(AggregateUnaryOperator aop, Future<FederatedResponse>[] ffr, FederationMap map) {
		if (aop.isRowAggregate() && map.getType() == FederationMap.FType.ROW)
			return bind(ffr, false);
		else if (aop.isColAggregate() && map.getType() == FederationMap.FType.COL)
			return bind(ffr, true);

		if (aop.aggOp.increOp.fn instanceof KahanFunction)
			return aggAdd(ffr);
		else if( aop.aggOp.increOp.fn instanceof Mean )
			return aggMean(ffr, map);
		else if (aop.aggOp.increOp.fn instanceof Builtin &&
			(((Builtin) aop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN ||
				((Builtin) aop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX)) {
			boolean isMin = ((Builtin) aop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN;
			return aggMinMax(ffr,isMin,false, Optional.of(map.getType()));
		} else
			throw new DMLRuntimeException("Unsupported aggregation operator: "
				+ aop.aggOp.increOp.fn.getClass().getSimpleName());
	}

	public static void waitFor(List<Future<FederatedResponse>> responses) {
		try {
			for(Future<FederatedResponse> fr : responses)
				fr.get();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static FederationMap federateLocalData(CacheableData<?> data) {
		long id = FederationUtils.getNextFedDataID();
		FederatedLocalData federatedLocalData = new FederatedLocalData(id, data);
		Map<FederatedRange, FederatedData> fedMap = new HashMap<>();
		fedMap.put(new FederatedRange(new long[2], new long[] {data.getNumRows(), data.getNumColumns()}),
			federatedLocalData);
		return new FederationMap(id, fedMap);
	}
}
