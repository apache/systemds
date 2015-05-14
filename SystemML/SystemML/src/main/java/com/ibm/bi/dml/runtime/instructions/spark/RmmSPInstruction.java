/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;


import java.util.LinkedList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.AggregateSumMultiBlockFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.TripleIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 */
public class RmmSPInstruction extends BinarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public RmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.RMM;		
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static RmmSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String opcode = InstructionUtils.getOpCode(str);

		if ( "rmm".equals(opcode) ) {
			String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			
			return new RmmSPInstruction(null, in1, in2, out, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("TsmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}		
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input rdds
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics( input1.getName() );
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics( input2.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
		
		//execute Spark RMM instruction
		//step 1: prepare join keys (w/ replication), i/j/k 
		JavaPairRDD<TripleIndexes,MatrixBlock> tmp1 = in1.flatMapToPair(
				new RmmReplicateFunction(mc2.getCols(), mc2.getColsPerBlock(), true)); 
		JavaPairRDD<TripleIndexes,MatrixBlock> tmp2 = in2.flatMapToPair(
				new RmmReplicateFunction(mc1.getRows(), mc1.getRowsPerBlock(), false));
		
		//step 2: join prepared datasets, multiply, and aggregate
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				tmp1.join( tmp2 )                                         //join by result block 
		            .mapToPair( new RmmMultiplyFunction() )               //do matrix multiplication
		            .reduceByKey( new AggregateSumMultiBlockFunction() ); //aggregation per result block
				
		//put output block into symbol table (no lineage because single block)
		updateOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}


	/**
	 * 
	 */
	private static class RmmReplicateFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, TripleIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 3577072668341033932L;
		
		private long _len = -1;
		private long _blen = -1;
		private boolean _left = false;
		
		public RmmReplicateFunction(long len, long blen, boolean left)
		{
			_len = len;
			_blen = blen;
			_left = left;
		}
		
		@Override
		public Iterable<Tuple2<TripleIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			LinkedList<Tuple2<TripleIndexes, MatrixBlock>> ret = new LinkedList<Tuple2<TripleIndexes, MatrixBlock>>();
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			long numBlocks = (long) Math.ceil((double)_len/_blen); 
			
			if( _left ) //LHS MATRIX
			{
				//replicate wrt # column blocks in RHS
				long i = ixIn.getRowIndex();
				long k = ixIn.getColumnIndex();
				for( long j=1; j<=numBlocks; j++ ) {
					TripleIndexes tmptix = new TripleIndexes(i, j, k);
					MatrixBlock tmpblk = new MatrixBlock(blkIn);
					ret.add( new Tuple2<TripleIndexes, MatrixBlock>(tmptix, tmpblk) );
				}
			} 
			else // RHS MATRIX
			{
				//replicate wrt # row blocks in LHS
				long k = ixIn.getRowIndex();
				long j = ixIn.getColumnIndex();
				for( long i=1; i<=numBlocks; i++ ) {
					TripleIndexes tmptix = new TripleIndexes(i, j, k);
					MatrixBlock tmpblk = new MatrixBlock(blkIn);
					ret.add( new Tuple2<TripleIndexes, MatrixBlock>(tmptix, tmpblk) );
				}
			}
			
			//output list of new tuples
			return ret;
		}
	}

	/**
	 * 
	 * 
	 */
	private static class RmmMultiplyFunction implements PairFunction<Tuple2<TripleIndexes, Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -5772410117511730911L;
		
		private AggregateBinaryOperator _op = null;
		
		public RmmMultiplyFunction()
		{
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<TripleIndexes, Tuple2<MatrixBlock,MatrixBlock>> arg0 ) 
			throws Exception 
		{
			//get input blocks per
			TripleIndexes ixIn = arg0._1(); //i,j,k
			MatrixIndexes ixOut = new MatrixIndexes(ixIn.getFirstIndex(), ixIn.getSecondIndex()); //i,j
			MatrixBlock blkIn1 = arg0._2()._1();
			MatrixBlock blkIn2 = arg0._2()._2();
			MatrixBlock blkOut = new MatrixBlock();
			
			//core block matrix multiplication 
			blkIn1.aggregateBinaryOperations(blkIn1, blkIn2, blkOut, _op);
							
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
}
