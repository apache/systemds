/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * TODO different aggregation types (single block / multi block via reduce/reducebykey)
 * 
 */
public class AggregateUnarySPInstruction extends UnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _aggregate = true;
	
	public AggregateUnarySPInstruction(AggregateUnaryOperator op, CPOperand in, CPOperand out, boolean aggregate, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_aggregate = aggregate;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		InstructionUtils.checkNumFields(str, 3);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		out.split(parts[2]);
		boolean agg = Boolean.parseBoolean(parts[3]);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		return new AggregateUnarySPInstruction(aggun, in1, out, agg, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		
		//execute unary aggregate
		AggregateUnaryOperator auop = (AggregateUnaryOperator) _optr;
		boolean dropCorrection = !_aggregate;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in.mapToPair(
				new RDDUAggFunction(auop, mc.getRowsPerBlock(), mc.getColsPerBlock(), dropCorrection));		
		if( _aggregate ) {
			out = out.reduceByKey(new RDDAggFunction(auop.aggOp));
			if( auop.aggOp.correctionExists )
				out = out.mapToPair(new RDDDropCorrectionFunction(auop.aggOp));
		}
		
		//put output handle in symbol table
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			throw new DMLRuntimeException("The output dimensions are not specified for AggregateUnarySPInstruction");
		}
		
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());
	}

	private static class RDDUAggFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 2672082409287856038L;
		
		private AggregateUnaryOperator _op = null;
		private int _brlen = -1;
		private int _bclen = -1;
		private boolean _dropCorr = false;
		
		public RDDUAggFunction( AggregateUnaryOperator op, int brlen, int bclen, boolean dropCorr )
		{
			_op = op;
			_brlen = brlen;
			_bclen = bclen;
			_dropCorr = dropCorr;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			//unary aggregate operation
			OperationsOnMatrixValues.performAggregateUnary( ixIn, blkIn, 
					  ixOut, blkOut, _op, _brlen, _bclen);
			if( _dropCorr )
				blkOut.dropLastRowsOrColums(_op.aggOp.correctionLocation);
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
	
	/**
	 * Note: currently we always include the correction and use a subsequent maptopair to
	 * drop them at the end because during aggregation we dont know if we produce an
	 * intermediate or the final aggregate. 
	 */
	public static class RDDAggFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = -3672377410407066396L;
	
		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;
		
		public RDDAggFunction( AggregateOperator op )
		{
			_op = op;	
			_corr = new MatrixBlock();
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//copy one first input
			MatrixBlock out = new MatrixBlock(arg0); 
			
			//aggregate second input
			_corr.reset(out.getNumRows(), out.getNumColumns());
			if(_op.correctionExists) {
				OperationsOnMatrixValues.incrementalAggregation(
						out, _corr, arg1, _op, true);
			}
			else {
				OperationsOnMatrixValues.incrementalAggregation(
						out, null, arg1, _op, true);
			}
			
			return out;
		}
	}
	
	public static class RDDDropCorrectionFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -5573656897943638857L;
		
		private AggregateOperator _op = null;
		
		public RDDDropCorrectionFunction(AggregateOperator op)
		{
			_op = op;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			//copy inputs
			MatrixIndexes ixOut = new MatrixIndexes(ixIn);
			MatrixBlock blkOut = new MatrixBlock(blkIn);
			
			//drop correction
			blkOut.dropLastRowsOrColums(_op.correctionLocation);
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
		
		
	}
}
