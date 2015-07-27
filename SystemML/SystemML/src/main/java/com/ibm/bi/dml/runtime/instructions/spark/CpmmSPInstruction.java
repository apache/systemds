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
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * Cpmm: cross-product matrix multiplication operation (distributed matrix multiply
 * by join over common dimension and subsequent aggregation of partial results).
 * 
 * NOTE: There is additional optimization potential by preventing aggregation for a single
 * block on the common dimension. However, in such a case we would never pick cpmm because
 * this would result in a degree of parallelism of 1.	
 * 
 */
public class CpmmSPInstruction extends BinarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private SparkAggType _aggtype;
	
	public CpmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, SparkAggType aggtype, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.CPMM;
		_aggtype = aggtype;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static CpmmSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( opcode.equalsIgnoreCase("cpmm")) {
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			SparkAggType aggtype = SparkAggType.valueOf(parts[4]);
			
			return new CpmmSPInstruction(aggbin, in1, in2, out, aggtype, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get rdd inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
		
		//process core cpmm matrix multiply 
		JavaPairRDD<Long, IndexedMatrixValue> tmp1 = in1.mapToPair(new CpmmIndexFunction(true));
		JavaPairRDD<Long, IndexedMatrixValue> tmp2 = in2.mapToPair(new CpmmIndexFunction(false));
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = tmp1
				   .join(tmp2)                              // join over common dimension
				   .mapToPair(new CpmmMultiplyFunction());  // compute block multiplications
				   
		//process cpmm aggregation and handle outputs				
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			MatrixBlock out2 = RDDAggregateUtils.sumStable(out);
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);	
		}
		else //DEFAULT: MULTI_BLOCK
		{
			out = RDDAggregateUtils.sumByKeyStable(out); 
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			sec.addLineageRDD(output.getName(), input2.getName());		
			
			//update output statistics if not inferred
			updateBinaryMMOutputMatrixCharacteristics(sec, true);
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class CpmmIndexFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, Long, IndexedMatrixValue>  
	{
		private static final long serialVersionUID = -1187183128301671162L;

		private boolean _left = false;
		
		public CpmmIndexFunction( boolean left ) {
			_left = left;
		}
		
		@Override
		public Tuple2<Long, IndexedMatrixValue> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			IndexedMatrixValue value = new IndexedMatrixValue();
			value.set(arg0._1(), new MatrixBlock(arg0._2()));
			
			Long key = _left ? arg0._1.getColumnIndex() : arg0._1.getRowIndex();
			return new Tuple2<Long, IndexedMatrixValue>(key, value);
		}	
	}

	/**
	 * 
	 *
	 */
	private static class CpmmMultiplyFunction implements PairFunction<Tuple2<Long, Tuple2<IndexedMatrixValue,IndexedMatrixValue>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -2009255629093036642L;
		
		private AggregateBinaryOperator _op = null;
		
		public CpmmMultiplyFunction()
		{
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<Long, Tuple2<IndexedMatrixValue, IndexedMatrixValue>> arg0)
			throws Exception 
		{
			MatrixBlock blkIn1 = (MatrixBlock)arg0._2()._1().getValue();
			MatrixBlock blkIn2 = (MatrixBlock)arg0._2()._2().getValue();
			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			//core block matrix multiplication 
			blkIn1.aggregateBinaryOperations(blkIn1, blkIn2, blkOut, _op);
			
			//return target block
			ixOut.setIndexes(arg0._2()._1().getIndexes().getRowIndex(), 
					         arg0._2()._2().getIndexes().getColumnIndex());
			return new Tuple2<MatrixIndexes, MatrixBlock>( ixOut, blkOut );
		}
	}
}
