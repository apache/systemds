/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.mr.BinaryInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;


public class BinUaggChainSPInstruction extends UnarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//operators
	private BinaryOperator _bOp = null;
	private AggregateUnaryOperator _uaggOp = null;
	
	public BinUaggChainSPInstruction(CPOperand in, CPOperand out, BinaryOperator bop, AggregateUnaryOperator uaggop, String opcode, String istr )
	{
		super(null, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.BinUaggChain;
		
		_bOp = bop;
		_uaggOp = uaggop;

	}

	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 4 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );		
		
		String opcode = parts[0];
		BinaryOperator bop = BinaryInstruction.parseBinaryOperator(parts[1]);
		AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[2]);
		in.split(parts[3]);
		out.split(parts[4]);
		
		return new BinUaggChainSPInstruction(in, out, bop, uaggop, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		
		//execute unary builtin operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				in.mapValues(new RDDBinUaggChainFunction(_bOp, _uaggOp));
		
		//set output RDD
		updateUnaryOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());
	}
	
	/**
	 * 
	 */
	public static class RDDBinUaggChainFunction implements Function<MatrixBlock,MatrixBlock> 
	{
		private static final long serialVersionUID = 886065328623752520L;
		
		private BinaryOperator _bOp = null;
		private AggregateUnaryOperator _uaggOp = null;
		
		public RDDBinUaggChainFunction(BinaryOperator bop, AggregateUnaryOperator uaggop) {
			_bOp = bop;
			_uaggOp = uaggop;
		}

		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			int brlen = arg0.getNumRows();
			int bclen = arg0.getNumColumns();
			
			//perform unary aggregate operation
			MatrixBlock out1 = new MatrixBlock();
			arg0.aggregateUnaryOperations(_uaggOp, out1, brlen, bclen, null);
			
			//strip-off correction
			out1.dropLastRowsOrColums(_uaggOp.aggOp.correctionLocation);
		
			//perform binary operation
			MatrixBlock out2 = new MatrixBlock();
			return (MatrixBlock) arg0.binaryOperations(_bOp, out1, out2);
		}		
	}
}
