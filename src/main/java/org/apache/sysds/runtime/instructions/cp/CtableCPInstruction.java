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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.Ctable;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.CTableMap;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap.EntryType;

public class CtableCPInstruction extends ComputationCPInstruction {
	private final CPOperand _outDim1;
	private final CPOperand _outDim2;
	private final boolean _isExpand;
	private final boolean _ignoreZeros;
	private final int _k;

	private CtableCPInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String outputDim1, boolean dim1Literal, String outputDim2, boolean dim2Literal, boolean isExpand,
			boolean ignoreZeros, String opcode, String istr, int k) {
		super(CPType.Ctable, null, in1, in2, in3, out, opcode, istr);
		_outDim1 = new CPOperand(outputDim1, ValueType.FP64, DataType.SCALAR, dim1Literal);
		_outDim2 = new CPOperand(outputDim2, ValueType.FP64, DataType.SCALAR, dim2Literal);
		_isExpand = isExpand;
		_ignoreZeros = ignoreZeros;
		_k = k;
	}

	public static CtableCPInstruction parseInstruction(String inst)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		InstructionUtils.checkNumFields ( parts, 8 );
		
		String opcode = parts[0];
		
		//handle opcode
		if ( !(opcode.equalsIgnoreCase(Opcodes.CTABLE.toString()) || opcode.equalsIgnoreCase(Opcodes.CTABLEEXPAND.toString())) ) {
			throw new DMLRuntimeException("Unexpected opcode in TertiaryCPInstruction: " + inst);
		}
		boolean isExpand = opcode.equalsIgnoreCase(Opcodes.CTABLEEXPAND.toString());
		
		//handle operands
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		
		//handle known dimension information
		String[] dim1Fields = parts[4].split(Instruction.LITERAL_PREFIX);
		String[] dim2Fields = parts[5].split(Instruction.LITERAL_PREFIX);

		CPOperand out = new CPOperand(parts[6]);
		boolean ignoreZeros = Boolean.parseBoolean(parts[7]);

		int k = Integer.parseInt(parts[8]);
		
		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new CtableCPInstruction(in1, in2, in3, out, 
			dim1Fields[0], Boolean.parseBoolean(dim1Fields[1]), dim2Fields[0],
			Boolean.parseBoolean(dim2Fields[1]), isExpand, ignoreZeros, opcode, inst, k);
	}

	private Ctable.OperationTypes findCtableOperation() {
		DataType dt1 = input1.getDataType();
		DataType dt2 = input2.getDataType();
		DataType dt3 = input3.getDataType();
		return Ctable.findCtableOperationByInputDataTypes(dt1, dt2, dt3);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixBlock matBlock1 = !_isExpand ? ec.getMatrixInput(input1): null;
		MatrixBlock matBlock2 = null, wtBlock=null;
		double cst1, cst2;
		
		CTableMap resultMap = new CTableMap(EntryType.INT);
		MatrixBlock resultBlock = null;
		Ctable.OperationTypes ctableOp = findCtableOperation();
		ctableOp = _isExpand ? Ctable.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT : ctableOp;
		
		long outputDim1 = ec.getScalarInput(_outDim1).getLongValue();
		long outputDim2 = ec.getScalarInput(_outDim2).getLongValue();
		
		boolean outputDimsKnown = (outputDim1 != -1 && outputDim2 != -1);
		if ( outputDimsKnown ) {
			int inputRows = matBlock1.getNumRows();
			int inputCols = matBlock1.getNumColumns();
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(outputDim1, outputDim2, inputRows*inputCols);
			//only create result block if dense; it is important not to aggregate on sparse result
			//blocks because it would implicitly turn the O(N) algorithm into O(N log N). 
			if( !sparse )
				resultBlock = new MatrixBlock((int)outputDim1, (int)outputDim2, false); 
		}
		
		switch(ctableOp) {
			case CTABLE_TRANSFORM: //(VECTOR)
				// F=ctable(A,B,W)
				matBlock2 = ec.getMatrixInput(input2.getName());
				wtBlock = ec.getMatrixInput(input3.getName());
				matBlock1.ctableOperations(_optr, matBlock2, wtBlock, resultMap, resultBlock);
				break;
			case CTABLE_TRANSFORM_SCALAR_WEIGHT: //(VECTOR/MATRIX)
				// F = ctable(A,B) or F = ctable(A,B,1)
				matBlock2 = ec.getMatrixInput(input2.getName());
				cst1 = ec.getScalarInput(input3).getDoubleValue();
				matBlock1.ctableOperations(_optr, matBlock2, cst1, _ignoreZeros, resultMap, resultBlock);
				break;
			case CTABLE_EXPAND_SCALAR_WEIGHT: //(VECTOR)
				// F = ctable(seq,A) or F = ctable(seq,B,1)
				// ignore first argument
				if(input1.getDataType() == DataType.MATRIX){
					LOG.warn("rewrite for table expand not activated please fix");
				}
				matBlock2 = ec.getMatrixInput(input2.getName());
				cst1 = ec.getScalarInput(input3).getDoubleValue();
				resultBlock = LibMatrixReorg.fusedSeqRexpand(matBlock2.getNumRows(), matBlock2, cst1, resultBlock, true, _k);
				break;
			case CTABLE_TRANSFORM_HISTOGRAM: //(VECTOR)
				// F=ctable(A,1) or F = ctable(A,1,1)
				cst1 = ec.getScalarInput(input2).getDoubleValue();
				cst2 = ec.getScalarInput(input3).getDoubleValue();
				matBlock1.ctableOperations(_optr, cst1, cst2, resultMap, resultBlock);
				break;
			case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: //(VECTOR)
				// F=ctable(A,1,W)
				wtBlock = ec.getMatrixInput(input3.getName());
				cst1 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral()).getDoubleValue();
				matBlock1.ctableOperations(_optr, cst1, wtBlock, resultMap, resultBlock);
				break;
			
			default:
				throw new DMLRuntimeException("Encountered an invalid ctable operation ("+ctableOp+") while executing instruction: " + this.toString());
		}
		
		if(input1.getDataType() == DataType.MATRIX && ctableOp != Ctable.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT)
			ec.releaseMatrixInput(input1.getName());
		if(input2.getDataType() == DataType.MATRIX)
			ec.releaseMatrixInput(input2.getName());
		if(input3.getDataType() == DataType.MATRIX)
			ec.releaseMatrixInput(input3.getName());
		
		if ( resultBlock == null ){
			//we need to respect potentially specified output dimensions here, because we might have 
			//decided for hash-aggregation just to prevent inefficiency in case of sparse outputs.  
			if( outputDimsKnown )
				resultBlock = DataConverter.convertToMatrixBlock( resultMap, (int)outputDim1, (int)outputDim2 );
			else
				resultBlock = DataConverter.convertToMatrixBlock( resultMap );
		}
		else
			resultBlock.examSparsity();
		
		// Ensure right dense/sparse output representation for special cases
		// such as ctable expand (guarded by released input memory)
		if( checkGuardedRepresentationChange(matBlock1, matBlock2, resultBlock) ) {
			resultBlock.examSparsity();
		}

		ec.setMatrixOutput(output.getName(), resultBlock);
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		LineageItem[] linputs = !(_outDim1.getName().equals("-1") && _outDim2.getName().equals("-1")) ?
			LineageItemUtils.getLineage(ec, input1, input2, input3, _outDim1, _outDim2) :
			LineageItemUtils.getLineage(ec, input1, input2, input3);
		return Pair.of(output.getName(), new LineageItem(getOpcode(), linputs));
	}

	public CPOperand getOutDim1() {
		return _outDim1;
	}

	public CPOperand getOutDim2() {
		return _outDim2;
	}

	public boolean getIsExpand() {
		return _isExpand;
	}

	public boolean getIgnoreZeros() {
		return _ignoreZeros;
	}
}
