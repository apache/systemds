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

package org.apache.sysml.runtime.instructions.mr;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class MRInstruction extends Instruction {

	public enum MRType {
		Append, Aggregate, Binary, Ternary, AggregateBinary, AggregateUnary, Rand,
		Seq, CSVReblock, CSVWrite, Reblock, Reorg, Replicate, Unary, CombineBinary, CombineUnary, CombineTernary,
		PickByCount, Partition, Ctable, Quaternary, CM_N_COV, MapGroupedAggregate, GroupedAggregate, RightIndex,
		ZeroOut, MMTSJ, PMMJ, MatrixReshape, ParameterizedBuiltin, Sort, MapMultChain, CumsumAggregate, CumsumSplit,
		CumsumOffset, BinUaggChain, UaggOuterChain, RemoveEmpty
	}

	protected final MRType mrtype;
	protected final Operator optr;
	public byte output;

	protected MRInstruction(MRType type, Operator op, byte out) {
		super.type = IType.MAPREDUCE;
		optr = op;
		output = out;
		mrtype = type;
	}

	public Operator getOperator() {
		return optr;
	}
	
	public MRType getMRInstructionType() {
		return mrtype;
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		//do nothing (not applicable for MR instructions)
	}

	public abstract void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor) 
		throws DMLRuntimeException;

	public abstract byte[] getInputIndexes() 
		throws DMLRuntimeException;

	public abstract byte[] getAllIndexes() 
		throws DMLRuntimeException;
}
