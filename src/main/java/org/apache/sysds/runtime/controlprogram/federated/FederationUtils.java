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
import java.util.concurrent.Future;

import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

public class FederationUtils {
	private static final IDSequence _idSeq = new IDSequence();
	
	public static long getNextFedDataID() {
		return _idSeq.getNextID();
	}
	
	public static FederatedRequest callInstruction(String inst, CPOperand varOldOut, CPOperand[] varOldIn, long[] varNewIn) {
		//TODO better and safe replacement of operand names --> instruction utils
		long id = getNextFedDataID();
		String linst = inst.replace(ExecType.SPARK.name(), ExecType.CP.name());
		linst = linst.replace(Lop.OPERAND_DELIMITOR+varOldOut.getName(), Lop.OPERAND_DELIMITOR+String.valueOf(id));
		for(int i=0; i<varOldIn.length; i++)
			if( varOldIn[i] != null )
				linst = linst.replace(Lop.OPERAND_DELIMITOR+varOldIn[i].getName(),
					Lop.OPERAND_DELIMITOR+String.valueOf(varNewIn[i]));
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

	public static MatrixBlock rbind(Future<FederatedResponse>[] ffr) {
		// TODO handle non-contiguous cases
		try {
			MatrixBlock[] tmp = getResults(ffr);
			return tmp[0].append(
				Arrays.copyOfRange(tmp, 1, tmp.length),
				new MatrixBlock(), false);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static ScalarObject aggScalar(AggregateUnaryOperator aop, Future<FederatedResponse>[] ffr) {
		if( !(aop.aggOp.increOp.fn instanceof KahanFunction) ) {
			throw new DMLRuntimeException("Unsupported aggregation operator: "
				+ aop.aggOp.increOp.getClass().getSimpleName());
		}
		//compute scalar sum of partial aggregates
		try {
			double sum = 0; //uak+, uasqk+
			for( Future<FederatedResponse> fr : ffr )
				sum += ((ScalarObject)fr.get().getData()[0]).getDoubleValue();
			return new DoubleObject(sum);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static MatrixBlock aggMatrix(AggregateUnaryOperator aop, Future<FederatedResponse>[] ffr) {
		if( !(aop.aggOp.increOp.fn instanceof KahanFunction) ) {
			throw new DMLRuntimeException("Unsupported aggregation operator: "
				+ aop.aggOp.increOp.getClass().getSimpleName());
		}
		
		//assumes full row partitions for row and col aggregates
		return aop.isRowAggregate() ?  rbind(ffr) : aggAdd(ffr);
	}
}
