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

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MergeBlocksFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * 
 */
public class CumulativeAggregateSPInstruction extends AggregateUnarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CumulativeAggregateSPInstruction(AggregateUnaryOperator op, CPOperand in1, CPOperand out, String opcode, String istr )
	{
		super(op, in1, out, null, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.CumsumAggregate;		
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static CumulativeAggregateSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );
		
		String opcode = parts[0];
		in1.split(parts[1]);
		out.split(parts[2]);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseCumulativeAggregateUnaryOperator(opcode);
		
		return new CumulativeAggregateSPInstruction(aggun, in1, out, opcode, str);	
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		long rlen = mc.getRows();
		int brlen = mc.getRowsPerBlock();
		int bclen = mc.getColsPerBlock();
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		
		//execute unary aggregate (w/ implicit drop correction)
		AggregateUnaryOperator auop = (AggregateUnaryOperator) _optr;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				in.mapToPair(new RDDCumAggFunction(auop, rlen, brlen, bclen))
				  .reduceByKey(new MergeBlocksFunction());  
		
		//put output handle in symbol table
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDCumAggFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 11324676268945117L;
		
		private AggregateUnaryOperator _op = null;		
		private long _rlen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		public RDDCumAggFunction( AggregateUnaryOperator op, long rlen, int brlen, int bclen )
		{
			_op = op;
			_rlen = rlen;
			_brlen = brlen;
			_bclen = bclen;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			//process instruction
			OperationsOnMatrixValues.performAggregateUnary( ixIn, blkIn, ixOut, blkOut, 
					                            ((AggregateUnaryOperator)_op), _brlen, _bclen);
			if( ((AggregateUnaryOperator)_op).aggOp.correctionExists )
				blkOut.dropLastRowsOrColums(((AggregateUnaryOperator)_op).aggOp.correctionLocation);
			
			//cumsum expand partial aggregates
			long rlenOut = (long)Math.ceil((double)_rlen/_brlen);
			long rixOut = (long)Math.ceil((double)ixIn.getRowIndex()/_brlen);
			int rlenBlk = (int) Math.min(rlenOut-(rixOut-1)*_brlen, _brlen);
			int clenBlk = blkOut.getNumColumns();
			int posBlk = (int) ((ixIn.getRowIndex()-1) % _brlen);
			MatrixBlock blkOut2 = new MatrixBlock(rlenBlk, clenBlk, false);
			blkOut2.copy(posBlk, posBlk, 0, clenBlk-1, blkOut, true);
			ixOut.setIndexes(rixOut, ixOut.getColumnIndex());
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut2);
		}
	}
}
