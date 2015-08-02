/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;

import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDSortUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;

public class QuantileSortSPInstruction extends UnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/*
	 * This class supports two variants of sort operation on a 1-dimensional input matrix. 
	 * The two variants are <code> weighted </code> and <code> unweighted </code>.
	 * Example instructions: 
	 *     sort:mVar1:mVar2 (input=mVar1, output=mVar2)
	 *     sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
	 *  
	 */
	
	public QuantileSortSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		this(op, in, null, out, opcode, istr);
	}
	
	public QuantileSortSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr){
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.QSort;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(SortKeys.OPCODE) ) {
			if ( parts.length == 3 ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				parseUnaryInstruction(str, in1, out);
				return new QuantileSortSPInstruction(new SimpleOperator(null), in1, out, opcode, str);
			}
			else if ( parts.length == 4 ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseUnaryInstruction(str, in1, in2, out);
				return new QuantileSortSPInstruction(new SimpleOperator(null), in1, in2, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a SortSPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		boolean weighted = (input2 != null);
		
		//get input rdds
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> inW = weighted ? sec.getBinaryBlockRDDHandleForVariable( input2.getName() ) : null;
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		long clen = -1;
		if( !weighted ) { //W/O WEIGHTS (default)
			out = RDDSortUtils.sortByVal(in, mc.getRows(), mc.getRowsPerBlock());
			clen = 1;
		}
		else { //W/ WEIGHTS
			out = RDDSortUtils.sortByVal(in, inW, mc.getRows(), mc.getRowsPerBlock());
			clen = 2;
		}

		//put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		if( weighted )
			sec.addLineageRDD(output.getName(), input2.getName());
		
		//update output matrix characteristics
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		mcOut.set(mc.getRows(), clen, mc.getRowsPerBlock(), mc.getColsPerBlock());
	}
}
