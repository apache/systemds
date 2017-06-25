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

package org.apache.sysml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;

import org.apache.sysml.lops.BinaryM.VectorType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.instructions.spark.functions.MatrixMatrixBinaryOpFunction;
import org.apache.sysml.runtime.instructions.spark.functions.MatrixScalarUnaryFunction;
import org.apache.sysml.runtime.instructions.spark.functions.MatrixVectorBinaryOpPartitionFunction;
import org.apache.sysml.runtime.instructions.spark.functions.OuterVectorBinaryOpFunction;
import org.apache.sysml.runtime.instructions.spark.functions.ReplicateVectorFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;

public abstract class BinarySPInstruction extends ComputationSPInstruction
{
	
	public BinarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr ){
		super(op, in1, in2, out, opcode, istr);
	}

	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out)
		throws DMLRuntimeException
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields ( parts, 3 );
		
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		
		return opcode;
	}
	
	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out)
		throws DMLRuntimeException
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields ( parts, 4 );
		
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		
		return opcode;
	}

	/**
	 * Common binary matrix-matrix process instruction
	 * 
	 * @param ec execution context
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected void processMatrixMatrixBinaryInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//sanity check dimensions
		checkMatrixMatrixBinaryCharacteristics(sec);
		
		// Get input RDDs
		String rddVar1 = input1.getName();
		String rddVar2 = input2.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( rddVar2 );
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics( rddVar1 );
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics( rddVar2 );
		
		BinaryOperator bop = (BinaryOperator) _optr;
	
		//vector replication if required (mv or outer operations)
		boolean rowvector = (mc2.getRows()==1 && mc1.getRows()>1);
		long numRepLeft = getNumReplicas(mc1, mc2, true);
		long numRepRight = getNumReplicas(mc1, mc2, false);
		if( numRepLeft > 1 )
			in1 = in1.flatMapToPair(new ReplicateVectorFunction(false, numRepLeft ));
		if( numRepRight > 1 )
			in2 = in2.flatMapToPair(new ReplicateVectorFunction(rowvector, numRepRight));
		
		//execute binary operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1
				.join(in2)
				.mapValues(new MatrixMatrixBinaryOpFunction(bop));
		
		//set output RDD
		updateBinaryOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar1);
		sec.addLineageRDD(output.getName(), rddVar2);
	}

	protected void processMatrixBVectorBinaryInstruction(ExecutionContext ec, VectorType vtype) 
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//sanity check dimensions
		checkMatrixMatrixBinaryCharacteristics(sec);

		//get input RDDs
		String rddVar = input1.getName(); 
		String bcastVar = input2.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
		PartitionedBroadcast<MatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar );
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(rddVar);
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(bcastVar);
		
		BinaryOperator bop = (BinaryOperator) _optr;
		boolean isOuter = (mc1.getRows()>1 && mc1.getCols()==1 && mc2.getRows()==1 && mc2.getCols()>1);
		
		//execute map binary operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( isOuter ) {
			out = in1.flatMapToPair(new OuterVectorBinaryOpFunction(bop, in2));
		}
		else { //default
			//note: we use mappartition in order to preserve partitioning information for
			//binary mv operations where the keys are guaranteed not to change, the reason
			//why we cannot use mapValues is the need for broadcast key lookups.
			//alternative: out = in1.mapToPair(new MatrixVectorBinaryOpFunction(bop, in2, vtype));
			out = in1.mapPartitionsToPair(
					new MatrixVectorBinaryOpPartitionFunction(bop, in2, vtype), true);
		}
		
		//set output RDD
		updateBinaryOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
		sec.addLineageBroadcast(output.getName(), bcastVar);
	}

	protected void processMatrixScalarBinaryInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
	
		//get input RDD
		String rddVar = (input1.getDataType() == DataType.MATRIX) ? input1.getName() : input2.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
		
		//get operator and scalar
		CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
		ScalarObject constant = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(), scalar.isLiteral());
		ScalarOperator sc_op = (ScalarOperator) _optr;
		sc_op.setConstant(constant.getDoubleValue());
		
		//execute scalar matrix arithmetic instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapValues( new MatrixScalarUnaryFunction(sc_op) );
			
		//put output RDD handle into symbol table
		updateUnaryOutputMatrixCharacteristics(sec, rddVar, output.getName());
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
	}

	protected MatrixCharacteristics updateBinaryMMOutputMatrixCharacteristics(SparkExecutionContext sec, boolean checkCommonDim) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) { 
			if( !mc1.dimsKnown() || !mc2.dimsKnown() )
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from inputs.");
			else if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() || mc1.getColsPerBlock() != mc2.getColsPerBlock())
				throw new DMLRuntimeException("Incompatible block sizes for BinarySPInstruction.");
			else if(checkCommonDim && mc1.getCols() != mc2.getRows())
				throw new DMLRuntimeException("Incompatible dimensions for BinarySPInstruction");
			else {
				mcOut.set(mc1.getRows(), mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
		}
		return mcOut;
	}

	protected void updateBinaryAppendOutputMatrixCharacteristics(SparkExecutionContext sec, boolean cbind) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		
		//infer initially unknown dimensions from inputs
		if(!mcOut.dimsKnown()) { 
			if( !mc1.dimsKnown() || !mc2.dimsKnown() )
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from inputs.");
			
			if( cbind )
				mcOut.set(mc1.getRows(), mc1.getCols()+mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			else //rbind
				mcOut.set(mc1.getRows()+mc2.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
		}	
		
		//infer initially unknown nnz from inputs
		if( !mcOut.nnzKnown() && mc1.nnzKnown() && mc2.nnzKnown() ) {
			mcOut.setNonZeros( mc1.getNonZeros() + mc2.getNonZeros() );
		}
	}

	protected long getNumReplicas(MatrixCharacteristics mc1, MatrixCharacteristics mc2, boolean left) 
	{
		if( left ) 
		{
			if(mc1.getCols()==1 ) //outer
				return (long) Math.ceil((double)mc2.getCols() / mc2.getColsPerBlock());	
		}
		else
		{
			if(mc2.getRows()==1 && mc1.getRows()>1) //outer, row vector
				return (long) Math.ceil((double)mc1.getRows() / mc1.getRowsPerBlock());	
			else if( mc2.getCols()==1 && mc1.getCols()>1 ) //col vector
				return (long) Math.ceil((double)mc1.getCols() / mc1.getColsPerBlock());			
		}
		
		return 1; //matrix-matrix
	}

	protected void checkMatrixMatrixBinaryCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException 
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		
		//check for unknown input dimensions
		if( !(mc1.dimsKnown() && mc2.dimsKnown()) ){
			throw new DMLRuntimeException("Unknown dimensions matrix-matrix binary operations: "
					+ "[" + mc1.getRows() + "x" + mc1.getCols()  + " vs " + mc2.getRows() + "x" + mc2.getCols() + "]");
		}
		
		//check for dimension mismatch
		if( (mc1.getRows() != mc2.getRows() ||  mc1.getCols() != mc2.getCols())
			&& !(mc1.getRows() == mc2.getRows() && mc2.getCols()==1 ) //matrix-colvector
			&& !(mc1.getCols() == mc2.getCols() && mc2.getRows()==1 ) //matrix-rowvector
			&& !(mc1.getCols()==1 && mc2.getRows()==1) )     //outer colvector-rowvector 
		{
			throw new DMLRuntimeException("Dimensions mismatch matrix-matrix binary operations: "
					+ "[" + mc1.getRows() + "x" + mc1.getCols()  + " vs " + mc2.getRows() + "x" + mc2.getCols() + "]");
		}	
		
		if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() ||  mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
			throw new DMLRuntimeException("Blocksize mismatch matrix-matrix binary operations: "
					+ "[" + mc1.getRowsPerBlock() + "x" + mc1.getColsPerBlock()  + " vs " + mc2.getRowsPerBlock() + "x" + mc2.getColsPerBlock() + "]");
		}	
	}

	protected void checkBinaryAppendInputCharacteristics(SparkExecutionContext sec, boolean cbind, boolean checkSingleBlk, boolean checkAligned) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		
		if(!mc1.dimsKnown() || !mc2.dimsKnown()) {
			throw new DMLRuntimeException("The dimensions unknown for inputs");
		}
		else if(cbind && mc1.getRows() != mc2.getRows()) {
			throw new DMLRuntimeException("The number of rows of inputs should match for append-cbind instruction");
		}
		else if(!cbind && mc1.getCols() != mc2.getCols()) {
			throw new DMLRuntimeException("The number of columns of inputs should match for append-rbind instruction");
		}
		else if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() || mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
			throw new DMLRuntimeException("The block sizes donot match for input matrices");
		}
		
		if( checkSingleBlk ) {
			if(mc1.getCols() + mc2.getCols() > mc1.getColsPerBlock())
				throw new DMLRuntimeException("Output must have at most one column block"); 
		}
		
		if( checkAligned ) {
			if( mc1.getCols() % mc1.getColsPerBlock() != 0 )
				throw new DMLRuntimeException("Input matrices are not aligned to blocksize boundaries. Wrong append selected");
		}
	}
}
