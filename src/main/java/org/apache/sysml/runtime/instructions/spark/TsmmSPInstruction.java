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

import org.apache.sysml.lops.MMTSJ.MMTSJType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;

/**
 * 
 */
public class TsmmSPInstruction extends UnarySPInstruction 
{
	
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
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		//check supported opcode 
		if ( !opcode.equalsIgnoreCase("tsmm") ) {
			throw new DMLRuntimeException("TsmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);			
		}
			
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType type = MMTSJType.valueOf(parts[3]);
		
		return new TsmmSPInstruction(null, in1, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
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
