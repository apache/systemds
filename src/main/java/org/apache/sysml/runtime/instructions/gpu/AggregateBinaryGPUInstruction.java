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
package org.apache.sysml.runtime.instructions.gpu;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.utils.Statistics;

public class AggregateBinaryGPUInstruction extends BinaryCPInstruction
{
	
	public AggregateBinaryGPUInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, 
			String opcode, String istr, boolean isLeftTransposed, boolean isRightTransposed){
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.AggregateBinary;
		this.isLeftTransposed = isLeftTransposed;
		this.isRightTransposed = isRightTransposed;
	}

	boolean isLeftTransposed;
	boolean isRightTransposed;
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static AggregateBinaryGPUInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( !opcode.equalsIgnoreCase("ba+*")) {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		InstructionUtils.checkNumFields( parts, 5 );
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		
		boolean isLeftTransposed = Boolean.parseBoolean(parts[4]);
		boolean isRightTransposed = Boolean.parseBoolean(parts[5]);
		
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, 1);
		return new AggregateBinaryGPUInstruction(aggbin, in1, in2, out, opcode, str, isLeftTransposed, isRightTransposed);	
	}
	
	private MatrixBlock transpose(MatrixBlock m1) throws DMLRuntimeException {
		ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), 1);
		return (MatrixBlock) (m1.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0));
	}
	
	private boolean isSparse(ExecutionContext ec, String var) throws DMLRuntimeException {
		MatrixObject mo = (MatrixObject) ec.getVariable(var);
		return LibMatrixCUDA.isInSparseFormat(mo);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{	
		// --------------------------------------
		// This code will be removed when the JIRA SYSTEMML-702 is complete
		if(	isSparse(ec, input1.getName()) || isSparse(ec, input2.getName())) {
			
			Statistics.gpuSparseMultCount.addAndGet(1);
			
			//get inputs
			MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
	        MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
	        
	        if(isLeftTransposed) 
	        	matBlock1 = transpose(matBlock1);
	        if(isRightTransposed) 
	        	matBlock2 = transpose(matBlock2);
			
	        //compute matrix multiplication
	        AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
			MatrixBlock soresBlock = (MatrixBlock) (matBlock1.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op));
				
			//release inputs/outputs
			ec.releaseMatrixInput(input1.getName());
			ec.releaseMatrixInput(input2.getName());
			ec.setMatrixOutput(output.getName(), soresBlock);
			return;
		}
		// --------------------------------------
		
		Statistics.incrementNoOfExecutedGPUInst();
		
		AggregateBinaryOperator op = (AggregateBinaryOperator) _optr;
		if( !(op.binaryFn instanceof Multiply && op.aggOp.increOp.fn instanceof Plus) ) {
			throw new DMLRuntimeException("Unsupported binary aggregate operation: ("+op.binaryFn+", "+op.aggOp+").");
		}
		
		//get inputs
		MatrixObject m1 = ec.getMatrixInputForGPUInstruction(input1.getName());
        MatrixObject m2 = ec.getMatrixInputForGPUInstruction(input2.getName());
        
        //compute matrix multiplication
        int rlen = (int) (isLeftTransposed ? m1.getNumColumns() : m1.getNumRows());
        int clen = (int) (isRightTransposed ? m2.getNumRows() : m2.getNumColumns());
        
        ec.setMetaData(output.getName(), rlen, clen);
        MatrixObject out = ec.getMatrixOutputForGPUInstruction(output.getName(), false);
        LibMatrixCUDA.matmult(m1, m2, out, isLeftTransposed, isRightTransposed);
        
		//release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(input1.getName());
		ec.releaseMatrixInputForGPUInstruction(input2.getName());
		ec.releaseMatrixOutputForGPUInstruction(output.getName());
	}
}
