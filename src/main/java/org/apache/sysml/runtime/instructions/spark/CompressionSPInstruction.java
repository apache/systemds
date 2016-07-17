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
import org.apache.spark.api.java.function.Function;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class CompressionSPInstruction extends UnarySPInstruction
{
	public CompressionSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.Reorg;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static CompressionSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields(str, 2);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);

		return new CompressionSPInstruction(null, in, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
	
		//get input rdd handle
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = 
				sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
	
		//execute compression
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				in.mapValues(new CompressionFunction());
			
		//set outputs
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(input1.getName(), output.getName());
	}
	
	/**
	 * 
	 */
	public static class CompressionFunction implements Function<MatrixBlock,MatrixBlock> 
	{	
		private static final long serialVersionUID = -6528833083609423922L;

		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			CompressedMatrixBlock cmb = new CompressedMatrixBlock(arg0);
			cmb.compress();
			
			return cmb;
		}
	}
}
