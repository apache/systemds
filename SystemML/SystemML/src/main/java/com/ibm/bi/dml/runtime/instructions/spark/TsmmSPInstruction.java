/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;


import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 */
public class TsmmSPInstruction extends UnarySPInstruction 
{
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
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String opcode = InstructionUtils.getOpCode(str);
		
		//check supported opcode 
		if ( !opcode.equalsIgnoreCase("tsmm") ) {
			throw new DMLRuntimeException("TsmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);			
		}
			
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		in1.split(parts[1]);
		out.split(parts[2]);
		MMTSJType type = MMTSJType.valueOf(parts[3]);
		
		return new TsmmSPInstruction(null, in1, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		
		//execute tsmm instruction (always produce exactly one output block)
		//(this formulation with values() requires --conf spark.driver.maxResultSize=0)
		RDDTSMMFunction ftsmm = new RDDTSMMFunction(_type);		
		JavaPairRDD<MatrixIndexes,MatrixBlock> tmp = in.mapValues(ftsmm);
		MatrixBlock out = RDDAggregateUtils.sumStable(tmp);
		      
		//put output block into symbol table (no lineage because single block)
		//this also includes implicit maintenance of matrix characteristics
		sec.setMatrixOutput(output.getName(), out);
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDTSMMFunction implements Function<MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 2935770425858019666L;
		
		private MMTSJType _type = null;
		
		public RDDTSMMFunction( MMTSJType type ) {
			_type = type;
		}
		
		@Override
		public MatrixBlock call( MatrixBlock arg0 ) 
			throws Exception 
		{
			//execute transpose-self matrix multiplication
			return arg0.transposeSelfMatrixMultOperations(new MatrixBlock(), _type);
		}
	}
	
}
