/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.instructions.spark.utils.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixReorg;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class MatrixReshapeSPInstruction extends UnarySPInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CPOperand _opRows = null;
	private CPOperand _opCols = null;
	private CPOperand _opByRow = null;
	
	public MatrixReshapeSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String istr)
	{
		super(op, in1, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MatrixReshape;
		
		_opRows = in2;
		_opCols = in3;
		_opByRow = in4;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand out = new CPOperand(parts[5]);
		 
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeInstruction: " + str);
		else
			return new MatrixReshapeSPInstruction(new Operator(true), in1, in2, in3, in4, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get parameters
		int rows = (int)ec.getScalarInput(_opRows.getName(), _opRows.getValueType(), _opRows.isLiteral()).getLongValue(); //save cast
		int cols = (int)ec.getScalarInput(_opCols.getName(), _opCols.getValueType(), _opCols.isLiteral()).getLongValue(); //save cast
		boolean byRow = ec.getScalarInput(_opByRow.getName(), ValueType.BOOLEAN, _opByRow.isLiteral()).getBooleanValue();
		
		//get inputs 
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics( input1.getName() );
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics( output.getName() );
		
		//update output characteristics and sanity check
		mcOut.set(rows, cols, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
		if( mcIn.getRows()*mcIn.getCols() != mcOut.getRows()*mcOut.getCols() ) {
			throw new DMLRuntimeException("Incompatible matrix characteristics for reshape: "
		                +mcIn.getRows()+"x"+mcIn.getCols()+" vs "+mcOut.getRows()+"x"+mcOut.getCols());
		}
		
		//execute reshape instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				in1.flatMapToPair(new RDDReshapeFunction(mcIn, mcOut, byRow));
		out = RDDAggregateUtils.mergeByKey(out);
		
		//put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
	}
	
	private static class RDDReshapeFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 2819309412002224478L;
		
		private MatrixCharacteristics _mcIn = null;
		private MatrixCharacteristics _mcOut = null;
		private boolean _byrow = true;
		
		public RDDReshapeFunction( MatrixCharacteristics mcIn, MatrixCharacteristics mcOut, boolean byrow)
		{
			_mcIn = mcIn;
			_mcOut = mcOut;
			_byrow = byrow;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			//input conversion (for libmatrixreorg compatibility)
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(arg0);
			
			//execute actual reshape operation
			ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();			
			out = LibMatrixReorg.reshape(in, _mcIn.getRows(), _mcIn.getCols(), _mcIn.getRowsPerBlock(), _mcIn.getRowsPerBlock(),
                    out, _mcOut.getRows(), _mcOut.getCols(), _mcOut.getRowsPerBlock(), _mcOut.getColsPerBlock(), _byrow);

			//output conversion (for compatibility w/ rdd schema)
			return SparkUtils.fromIndexedMatrixBlock(out);
		}
	}
}
