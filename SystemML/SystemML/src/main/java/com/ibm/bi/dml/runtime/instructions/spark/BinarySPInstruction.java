/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.broadcast.Broadcast;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MatrixMatrixBinaryOpFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MatrixScalarUnaryFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MatrixVectorBinaryOpFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ReplicateVectorFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;

public abstract class BinarySPInstruction extends ComputationSPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BinarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr ){
		super(op, in1, in2, out, opcode, istr);
	}

	public BinarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr ){
		super(op, in1, in2, in3, out, opcode, istr);
	}
	
	/**
	 * 
	 * @param instr
	 * @param in1
	 * @param in2
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out)
		throws DMLRuntimeException
	{	
		InstructionUtils.checkNumFields ( instr, 3 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		
		return opcode;
	}
	
	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out)
		throws DMLRuntimeException
	{
		InstructionUtils.checkNumFields ( instr, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
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
	 * @param ec
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	protected void processMatrixMatrixBinaryInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
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
	
	/**
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void processMatrixBVectorBinaryInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//sanity check dimensions
		checkMatrixMatrixBinaryCharacteristics(sec);

		//get input RDDs
		String rddVar = input1.getName(); 
		String bcastVar = input2.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
		Broadcast<PartitionedMatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar );
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(rddVar);
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(bcastVar);
		
		BinaryOperator bop = (BinaryOperator) _optr;
		boolean isColVector = (mc2.getCols() == 1);
		boolean isOuter = (mc1.getCols() == 1 && mc2.getRows() == 1);
		
		//execute map binary operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1
				.flatMapToPair(new MatrixVectorBinaryOpFunction(bop, in2, true, isColVector, isOuter));
		
		//set output RDD
		updateBinaryOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
		sec.addLineageBroadcast(output.getName(), bcastVar);
	}
	
	/**
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void processMatrixScalarBinaryInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
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
	
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	protected void updateBinaryMMOutputMatrixCharacteristics(SparkExecutionContext sec, boolean checkCommonDim) 
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
	}
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	protected void updateBinaryAppendOutputMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) { 
			if( !mc1.dimsKnown() || !mc2.dimsKnown() )
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from inputs.");
			
			mcOut.set(mc1.getRows(), mc1.getCols()+mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
		}	
	}

	/**
	 * 
	 * @param mc1
	 * @param mc2
	 * @param left
	 * @return
	 */
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
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	protected void checkMatrixMatrixBinaryCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException 
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		
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
}
