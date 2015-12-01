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

import org.apache.spark.api.java.JavaPairRDD;

import org.apache.sysml.lops.SortKeys;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.utils.RDDSortUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;

public class QuantileSortSPInstruction extends UnarySPInstruction
{
	
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
