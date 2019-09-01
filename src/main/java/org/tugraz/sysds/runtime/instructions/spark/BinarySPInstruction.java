/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.lops.BinaryM.VectorType;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.tugraz.sysds.runtime.instructions.spark.functions.MatrixMatrixBinaryOpFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.MatrixScalarUnaryFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.MatrixVectorBinaryOpPartitionFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.OuterVectorBinaryOpFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.ReplicateTensorFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.ReplicateVectorFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.TensorTensorBinaryOpFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.TensorTensorBinaryOpPartitionFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.ScalarOperator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

public abstract class BinarySPInstruction extends ComputationSPInstruction {

	protected BinarySPInstruction(SPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}
	
	public static BinarySPInstruction parseInstruction ( String str ) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = null;
		boolean isBroadcast = false;
		VectorType vtype = null;
		
		if(str.startsWith("SPARK"+Lop.OPERAND_DELIMITOR+"map")) {
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			InstructionUtils.checkNumFields ( parts, 5 );
			
			opcode = parts[0];
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			vtype = VectorType.valueOf(parts[5]);
			isBroadcast = true;
		}
		else {
			opcode = parseBinaryInstruction(str, in1, in2, out);
		}
		
		DataType dt1 = in1.getDataType();
		DataType dt2 = in2.getDataType();
		
		Operator operator = InstructionUtils.parseExtendedBinaryOrBuiltinOperator(opcode, in1, in2);
		
		if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX) {
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX) {
				if(isBroadcast)
					return new BinaryMatrixBVectorSPInstruction(operator, in1, in2, out, vtype, opcode, str);
				else
					return new BinaryMatrixMatrixSPInstruction(operator, in1, in2, out, opcode, str);
			}
			else
				return new BinaryMatrixScalarSPInstruction(operator, in1, in2, out, opcode, str);
		}
		else if (dt1 == DataType.TENSOR || dt2 == DataType.TENSOR) {
			if (dt1 == DataType.TENSOR && dt2 == DataType.TENSOR) {
				if (isBroadcast)
					return new BinaryTensorTensorBroadcastSPInstruction(operator, in1, in2, out, opcode, str);
				else
					return new BinaryTensorTensorSPInstruction(operator, in1, in2, out, opcode, str);
			}
			else
				throw new DMLRuntimeException("Tensor binary operation not yet implemented for tensor-scalar, or tensor-matrix");
		}
		return null;
	}

	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields ( parts, 3 );
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		return opcode;
	}
	
	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) {
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
	 */
	protected void processMatrixMatrixBinaryInstruction(ExecutionContext ec) 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//sanity check dimensions
		checkMatrixMatrixBinaryCharacteristics(sec);
		updateBinaryOutputDataCharacteristics(sec);
		
		// Get input RDDs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable(input2.getName());
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		
		BinaryOperator bop = (BinaryOperator) _optr;
	
		//vector replication if required (mv or outer operations)
		boolean rowvector = (mc2.getRows()==1 && mc1.getRows()>1);
		long numRepLeft = getNumReplicas(mc1, mc2, true);
		long numRepRight = getNumReplicas(mc1, mc2, false);
		if( numRepLeft > 1 )
			in1 = in1.flatMapToPair(new ReplicateVectorFunction(false, numRepLeft ));
		if( numRepRight > 1 )
			in2 = in2.flatMapToPair(new ReplicateVectorFunction(rowvector, numRepRight));
		int numPrefPart = SparkUtils.isHashPartitioned(in1) ? in1.getNumPartitions() :
			SparkUtils.isHashPartitioned(in2) ? in2.getNumPartitions() :
			Math.min(in1.getNumPartitions() + in2.getNumPartitions(),
				2 * SparkUtils.getNumPreferredPartitions(mcOut));
		
		//execute binary operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1
			.join(in2, numPrefPart)
			.mapValues(new MatrixMatrixBinaryOpFunction(bop));
		
		//set output RDD
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}

	/**
	 * Common binary tensor-tensor process instruction
	 *
	 * @param ec execution context
	 */
	protected void processTensorTensorBinaryInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;

		//sanity check dimensions
		checkTensorTensorBinaryCharacteristics(sec);
		updateBinaryTensorOutputDataCharacteristics(sec);

		// Get input RDDs
		JavaPairRDD<TensorIndexes, TensorBlock> in1 = sec.getBinaryTensorBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<TensorIndexes, TensorBlock> in2 = sec.getBinaryTensorBlockRDDHandleForVariable(input2.getName());
		DataCharacteristics tc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics tc2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics dcOut = sec.getDataCharacteristics(output.getName());

		BinaryOperator bop = (BinaryOperator) _optr;

		// TODO blocking scheme for tensors/matrices with mismatching number of dimensions
		for (int i = 0; i < tc1.getNumDims(); i++) {
			long numReps = getNumDimReplicas(tc1, tc2, i);
			if (numReps > 1)
				in2 = in2.flatMapToPair(new ReplicateTensorFunction(i, numReps));
		}
		int numPrefPart = SparkUtils.isHashPartitioned(in1) ? in1.getNumPartitions() :
				SparkUtils.isHashPartitioned(in2) ? in2.getNumPartitions() :
						Math.min(in1.getNumPartitions() + in2.getNumPartitions(),
								2 * SparkUtils.getNumPreferredPartitions(dcOut));

		//execute binary operation
		JavaPairRDD<TensorIndexes, TensorBlock> out = in1
				.join(in2, numPrefPart)
				.mapValues(new TensorTensorBinaryOpFunction(bop));

		//set output RDD
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}

	protected void processMatrixBVectorBinaryInstruction(ExecutionContext ec, VectorType vtype)
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//sanity check dimensions
		checkMatrixMatrixBinaryCharacteristics(sec);

		//get input RDDs
		String rddVar = input1.getName(); 
		String bcastVar = input2.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( rddVar );
		PartitionedBroadcast<MatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar );
		DataCharacteristics mc1 = sec.getDataCharacteristics(rddVar);
		DataCharacteristics mc2 = sec.getDataCharacteristics(bcastVar);
		
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
		updateBinaryOutputDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
		sec.addLineageBroadcast(output.getName(), bcastVar);
	}

	protected void processTensorTensorBroadcastBinaryInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		//sanity check dimensions
		checkTensorTensorBinaryCharacteristics(sec);

		//get input RDDs
		String rddVar = input1.getName();
		String bcastVar = input2.getName();
		JavaPairRDD<TensorIndexes, TensorBlock> in1 = sec.getBinaryTensorBlockRDDHandleForVariable(rddVar);
		PartitionedBroadcast<TensorBlock> in2 = sec.getBroadcastForTensorVariable(bcastVar);
		DataCharacteristics mc2 = sec.getDataCharacteristics(bcastVar);

		BinaryOperator bop = (BinaryOperator) _optr;

		boolean[] replicateDim = new boolean[mc2.getNumDims()];
		for (int i = 0; i < replicateDim.length; i++)
			replicateDim[i] = mc2.getDim(i) == 1;

		//execute map binary operation
		JavaPairRDD<TensorIndexes, TensorBlock> out = null;
		out = in1.mapPartitionsToPair(
				new TensorTensorBinaryOpPartitionFunction(bop, in2, replicateDim), true);

		//set output RDD
		updateBinaryTensorOutputDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
		sec.addLineageBroadcast(output.getName(), bcastVar);
	}

	protected void processMatrixScalarBinaryInstruction(ExecutionContext ec)
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
	
		//get input RDD
		String rddVar = (input1.getDataType() == DataType.MATRIX) ? input1.getName() : input2.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( rddVar );
		
		//get operator and scalar
		CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
		ScalarObject constant = (ScalarObject) ec.getScalarInput(scalar);
		ScalarOperator sc_op = (ScalarOperator) _optr;
		sc_op = sc_op.setConstant(constant.getDoubleValue());
		
		//execute scalar matrix arithmetic instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapValues( new MatrixScalarUnaryFunction(sc_op) );
			
		//put output RDD handle into symbol table
		updateUnaryOutputDataCharacteristics(sec, rddVar, output.getName());
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
	}

	protected DataCharacteristics updateBinaryMMOutputDataCharacteristics(SparkExecutionContext sec, boolean checkCommonDim)
	{
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) { 
			if( !mc1.dimsKnown() || !mc2.dimsKnown() )
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from inputs.");
			else if(mc1.getBlocksize() != mc2.getBlocksize() || mc1.getBlocksize() != mc2.getBlocksize())
				throw new DMLRuntimeException("Incompatible block sizes for BinarySPInstruction.");
			else if(checkCommonDim && mc1.getCols() != mc2.getRows())
				throw new DMLRuntimeException("Incompatible dimensions for BinarySPInstruction");
			else {
				mcOut.set(mc1.getRows(), mc2.getCols(), mc1.getBlocksize(), mc1.getBlocksize());
			}
		}
		return mcOut;
	}

	protected void updateBinaryAppendOutputDataCharacteristics(SparkExecutionContext sec, boolean cbind)
	{
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		
		//infer initially unknown dimensions from inputs
		if(!mcOut.dimsKnown()) { 
			if( !mc1.dimsKnown() || !mc2.dimsKnown() )
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from inputs.");
			
			if( cbind )
				mcOut.set(mc1.getRows(), mc1.getCols()+mc2.getCols(), mc1.getBlocksize(), mc1.getBlocksize());
			else //rbind
				mcOut.set(mc1.getRows()+mc2.getRows(), mc1.getCols(), mc1.getBlocksize(), mc1.getBlocksize());
		}	
		
		//infer initially unknown nnz from inputs
		if( !mcOut.nnzKnown() && mc1.nnzKnown() && mc2.nnzKnown() ) {
			mcOut.setNonZeros( mc1.getNonZeros() + mc2.getNonZeros() );
		}
	}

	protected long getNumReplicas(DataCharacteristics mc1, DataCharacteristics mc2, boolean left)
	{
		if( left ) 
		{
			if(mc1.getCols()==1 ) //outer
				return mc2.getNumColBlocks();
		}
		else
		{
			if(mc2.getRows()==1 && mc1.getRows()>1) //outer, row vector
				return mc1.getNumRowBlocks();
			else if( mc2.getCols()==1 && mc1.getCols()>1 ) //col vector
				return mc2.getNumColBlocks();
		}
		
		return 1; //matrix-matrix
	}

	protected long getNumDimReplicas(DataCharacteristics dc1, DataCharacteristics dc2, int dim) {
		if (dc2.getDim(dim) == 1 && dc2.getDim(dim) > 1)
			return dc1.getNumBlocks(dim);
		return 1;
	}

	protected void checkMatrixMatrixBinaryCharacteristics(SparkExecutionContext sec) 
	{
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		
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
		
		if(mc1.getBlocksize() != mc2.getBlocksize() ||  mc1.getBlocksize() != mc2.getBlocksize()) {
			throw new DMLRuntimeException("Blocksize mismatch matrix-matrix binary operations: "
					+ "[" + mc1.getBlocksize() + "x" + mc1.getBlocksize()  + " vs " + mc2.getBlocksize() + "x" + mc2.getBlocksize() + "]");
		}	
	}

	protected void checkTensorTensorBinaryCharacteristics(SparkExecutionContext sec)
	{
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());

		//check for unknown input dimensions
		if (!(mc1.dimsKnown() && mc2.dimsKnown())) {
			// TODO print dimensions
			throw new DMLRuntimeException("Unknown dimensions tensor-tensor binary operations");
		}

		boolean dimensionMismatch = mc1.getNumDims() != mc2.getNumDims();
		if (!dimensionMismatch) {
			for (int i = 0; i < mc1.getNumDims(); i++) {
				if (mc1.getDim(i) != mc2.getDim(i) && mc2.getDim(i) != 1) {
					dimensionMismatch = true;
					break;
				}
			}
		}
		//check for dimension mismatch
		if (dimensionMismatch) {
			throw new DMLRuntimeException("Dimensions mismatch tensor-tensor binary operations");
		}

		boolean blockSizeMismatch = false;
		for (int i = 0; i < mc1.getNumDims(); i++) {
			if (mc1.getNumBlocks(i) != mc2.getNumBlocks(i) && mc2.getDim(i) != 1) {
				blockSizeMismatch = true;
				break;
			}
		}
		if (blockSizeMismatch)
			throw new DMLRuntimeException("Blocksize mismatch tensor-tensor binary operations");
	}

	protected void checkBinaryAppendInputCharacteristics(SparkExecutionContext sec, boolean cbind, boolean checkSingleBlk, boolean checkAligned)
	{
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		
		if(!mc1.dimsKnown() || !mc2.dimsKnown()) {
			throw new DMLRuntimeException("The dimensions unknown for inputs");
		}
		else if(cbind && mc1.getRows() != mc2.getRows()) {
			throw new DMLRuntimeException("The number of rows of inputs should match for append-cbind instruction");
		}
		else if(!cbind && mc1.getCols() != mc2.getCols()) {
			throw new DMLRuntimeException("The number of columns of inputs should match for append-rbind instruction");
		}
		else if(mc1.getBlocksize() != mc2.getBlocksize() || mc1.getBlocksize() != mc2.getBlocksize()) {
			throw new DMLRuntimeException("The block sizes donot match for input matrices");
		}
		
		if( checkSingleBlk ) {
			if(mc1.getCols() + mc2.getCols() > mc1.getBlocksize())
				throw new DMLRuntimeException("Output must have at most one column block"); 
		}
		
		if( checkAligned ) {
			if( mc1.getCols() % mc1.getBlocksize() != 0 )
				throw new DMLRuntimeException("Input matrices are not aligned to blocksize boundaries. Wrong append selected");
		}
	}
}
