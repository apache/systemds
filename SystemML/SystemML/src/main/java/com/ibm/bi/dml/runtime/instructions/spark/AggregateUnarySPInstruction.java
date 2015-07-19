/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.hops.AggBinaryOp.SparkAggType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.AggregateDropCorrectionFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.AggregateMultiBlockFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.AggregateSingleBlockFunction;
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
	
	private SparkAggType _aggtype;
	
	public AggregateUnarySPInstruction(AggregateUnaryOperator op, CPOperand in, CPOperand out, SparkAggType aggtype, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_aggtype = aggtype;
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
		SparkAggType aggtype = SparkAggType.valueOf(parts[3]);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		return new AggregateUnarySPInstruction(aggun, in1, out, aggtype, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		boolean aggregate = (_aggtype != SparkAggType.NONE);
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		
		//execute unary aggregate operation
		AggregateUnaryOperator auop = (AggregateUnaryOperator)_optr;
		AggregateOperator aggop = auop.aggOp;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in.mapToPair(
				new RDDUAggFunction(auop, mc.getRowsPerBlock(), mc.getColsPerBlock(), !aggregate));		
		
		//perform aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			MatrixBlock out2 = out.values()
					.reduce(new AggregateSingleBlockFunction(aggop));
			
			//drop correction after aggregation
			out2.dropLastRowsOrColums(aggop.correctionLocation);
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);
		}
		else //MULTI_BLOCK or NONE
		{
			if( _aggtype == SparkAggType.MULTI_BLOCK ) {
				out = out.reduceByKey( new AggregateMultiBlockFunction(aggop) );
	
				//drop correction after aggregation
				if( auop.aggOp.correctionExists ) {
					out = out.mapToPair( new AggregateDropCorrectionFunction(aggop) );
				}
			}
			
			//put output RDD handle into symbol table
			checkExistingOutputDimensions(sec);
			sec.setRDDHandleForVariable(output.getName(), out);	
			sec.addLineageRDD(output.getName(), input1.getName());
		}		
	}

	/**
	 * 
	 */
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
			
			//drop correction if necessary
			if( _dropCorr ) {
				blkOut.dropLastRowsOrColums(_op.aggOp.correctionLocation);
			}
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
}
