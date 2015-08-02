/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MatrixMatrixBinaryOpFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ReplicateVectorFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class MatrixMatrixArithmeticSPInstruction extends ArithmeticBinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixArithmeticSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) 
		throws DMLRuntimeException
	{
		super(op, in1, in2, out, opcode, istr);
		
		//sanity check opcodes
		if ( !(  opcode.equalsIgnoreCase("+") || opcode.equalsIgnoreCase("-") || opcode.equalsIgnoreCase("*")
			  || opcode.equalsIgnoreCase("/") || opcode.equalsIgnoreCase("%%") || opcode.equalsIgnoreCase("%/%")
			  || opcode.equalsIgnoreCase("^")) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in MatrixMatrixArithmeticSPInstruction: " + toString());
		}		
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
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
	
}