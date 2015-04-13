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

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 */
public class TsmmSPInstruction extends UnarySPInstruction {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MMTSJType _type = null;
	
	public TsmmSPInstruction(Operator op, CPOperand in1, CPOperand out, MMTSJType type, String opcode, String istr )
	{
		super(op, in1, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.TSMM;		
		_type = type;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static TsmmSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String opcode = InstructionUtils.getOpCode(str);

		if ( "tsmm".equals(opcode) ) {
			String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
			in1.split(parts[1]);
			out.split(parts[2]);
			MMTSJType type = MMTSJType.valueOf(parts[3]);
			
			return new TsmmSPInstruction(null, in1, out, type, opcode, str);
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
		String opcode = getOpcode();
		
		if ( "tsmm".equals(opcode) )
		{
			//get input
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getRDDHandleForVariable( input1.getName() );
			
			//execute tsmm instruction
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
					in.mapToPair( new RDDTSMMFunction(_type) )
			          .reduceByKey( new RDDTSMMAggregateSumFunction() );
			
			//put output RDD handle into symbol table
			updateOutputMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
		}
		else 
		{
			throw new DMLRuntimeException("Unknown opcode: " + toString());
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDTSMMFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 2935770425858019666L;
		
		private MMTSJType _type = null;
		
		public RDDTSMMFunction( MMTSJType type )
		{
			_type = type;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes(1,1);
			MatrixBlock blkOut = new MatrixBlock();
			
			//execute matrix-vector mult
			blkIn.transposeSelfMatrixMultOperations(blkOut, _type);
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
	
	/**
	 *
	 */
	public static class RDDTSMMAggregateSumFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 704959820141782708L;
		
		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;
		
		public RDDTSMMAggregateSumFunction()
		{
			_op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);	
			_corr = null;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//create correction block (on demand)
			if( _corr == null ){
				_corr = new MatrixBlock(arg0.getNumRows(), arg0.getNumColumns(), true);
			}
			
			//copy one input to output
			MatrixBlock out = new MatrixBlock(arg0);
			
			//aggregate other input
			OperationsOnMatrixValues.incrementalAggregation(out, _corr, arg1, _op, false);
			
			return out;
		}
	}
}
