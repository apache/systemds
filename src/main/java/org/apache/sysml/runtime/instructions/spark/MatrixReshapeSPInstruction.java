/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class MatrixReshapeSPInstruction extends UnarySPInstruction
{	
	
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
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields( parts, 5 );
		
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
