/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class MRInstruction extends Instruction 
{
	
	public enum MRINSTRUCTION_TYPE { INVALID, Append, Aggregate, ArithmeticBinary, AggregateBinary, AggregateUnary, 
		Rand, Seq, CSVReblock, CSVWrite, Transform,
		Reblock, Reorg, Replicate, Unary, CombineBinary, CombineUnary, CombineTernary, PickByCount, Partition,
		Ternary, Quaternary, CM_N_COV, Combine, GroupedAggregate, RangeReIndex, ZeroOut, MMTSJ, PMMJ, MatrixReshape, ParameterizedBuiltin, Sort, MapMultChain,
		CumsumAggregate, CumsumSplit, CumsumOffset, BinUaggChain, UaggOuterChain, RemoveEmpty}; 
	
	
	protected MRINSTRUCTION_TYPE mrtype;
	protected Operator optr;
	public byte output;
	
	public MRInstruction (Operator op, byte out) {
		type = INSTRUCTION_TYPE.MAPREDUCE;
		optr = op;
		output = out;
		mrtype = MRINSTRUCTION_TYPE.INVALID;
	}

	public Operator getOperator() {
		return optr;
	}
	
	public MRINSTRUCTION_TYPE getMRInstructionType() 
	{
		return mrtype;
	}

	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//do nothing (not applicable for MR instructions)
	}

	/**
	 * 
	 * @param valueClass
	 * @param cachedValues
	 * @param tempValue
	 * @param zeroInput
	 * @param blockRowFactor
	 * @param blockColFactor
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public abstract void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor) 
		throws DMLUnsupportedOperationException, DMLRuntimeException;

	
	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public abstract byte[] getInputIndexes() 
		throws DMLRuntimeException;

	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public abstract byte[] getAllIndexes() 
		throws DMLRuntimeException;
}
