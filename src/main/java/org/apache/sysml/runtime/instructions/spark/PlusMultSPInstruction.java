/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.MinusMultiply;
import org.apache.sysml.runtime.functionobjects.PlusMultiply;
import org.apache.sysml.runtime.functionobjects.ValueFunctionWithConstant;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.PlusMultCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.spark.functions.MatrixMatrixBinaryOpFunction;
import org.apache.sysml.runtime.instructions.spark.functions.ReplicateVectorFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class PlusMultSPInstruction extends  ArithmeticBinarySPInstruction
{
	public PlusMultSPInstruction(BinaryOperator op, CPOperand in1, CPOperand in2, 
			CPOperand in3, CPOperand out, String opcode, String str) throws DMLRuntimeException 
	{
		super(op, in1, in2, out, opcode, str);
		input3= in3;
		
		//sanity check opcodes
		if ( !(  opcode.equalsIgnoreCase("+*") || opcode.equalsIgnoreCase("-*")  ) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in PlusMultSPInstruction: " + toString());
		}		
	}
	public static PlusMultSPInstruction parseInstruction(String str) throws DMLRuntimeException
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode=parts[0];
		CPOperand operand1 = new CPOperand(parts[1]);
		CPOperand operand2 = new CPOperand(parts[2]);
		CPOperand operand3 = new CPOperand(parts[3]);
		CPOperand outOperand = new CPOperand(parts[4]);
		BinaryOperator bOperator = null;
		if(opcode.equals("+*"))
			bOperator = new BinaryOperator(new PlusMultiply());
		else if (opcode.equals("-*"))
			bOperator = new BinaryOperator(new MinusMultiply());
		return new PlusMultSPInstruction(bOperator,operand1, operand2, operand3, outOperand, opcode,str);	
	}
	
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//sanity check dimensions
		checkMatrixMatrixBinaryCharacteristics(sec);
		
		// Get input RDDs
		String rddVar1 = input1.getName();
		String rddVar3 = input3.getName();
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = sec.getBinaryBlockRDDHandleForVariable( rddVar3 );
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics( rddVar1 );
		MatrixCharacteristics mc3 = sec.getMatrixCharacteristics( rddVar3 );
		
		ScalarObject constant = (ScalarObject) ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral());
		((ValueFunctionWithConstant) ((BinaryOperator)_optr).fn).setConstant(constant.getDoubleValue());

		BinaryOperator bop = (BinaryOperator) _optr;
		//vector replication if required (mv or outer operations)
		boolean rowvector = (mc3.getRows()==1 && mc1.getRows()>1);
		long numRepLeft = getNumReplicas(mc1, mc3, true);
		long numRepRight = getNumReplicas(mc1, mc3, false);
		if( numRepLeft > 1 )
			in1 = in1.flatMapToPair(new ReplicateVectorFunction(false, numRepLeft ));
		if( numRepRight > 1 )
			in3 = in3.flatMapToPair(new ReplicateVectorFunction(rowvector, numRepRight));
		
		//execute operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1
				.join(in3)
				.mapValues(new MatrixMatrixBinaryOpFunction(bop));
		
		//set output RDD
		updateBinaryOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar1);
		sec.addLineageRDD(output.getName(), rddVar3);
	}
	
}