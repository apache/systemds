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

import scala.Tuple2;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;

/**
 * 
 */
public class AggregateTernarySPInstruction extends ComputationSPInstruction
{
	
	public AggregateTernarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, 
			                             CPOperand out, String opcode, String istr )
	{
		super(op, in1, in2, in3, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.AggregateTernary;
	}

	public static AggregateTernarySPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("tak+*")) {
			InstructionUtils.checkNumFields ( parts, 4 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);

			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject());
			AggregateBinaryOperator op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			
			return new AggregateTernarySPInstruction(op, in1, in2, in3, out, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("AggregateTertiaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = input3.isLiteral() ? null : //matrix or literal 1
													 sec.getBinaryBlockRDDHandleForVariable( input3.getName() );
		
		//execute aggregate ternary operation
		AggregateBinaryOperator aggop = (AggregateBinaryOperator) _optr;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( in3 != null ) { //3 inputs
			out = in1.join( in2 ).join( in3 )
				     .mapValues(new RDDAggregateTernaryFunction(aggop));
		}
		else { //2 inputs (third is literal 1)
			out = in1.join( in2 )
					 .mapValues(new RDDAggregateTernaryFunction2(aggop));				
		}
				
		//aggregate and create output (no lineage because scalar)	   
		MatrixBlock tmp = RDDAggregateUtils.sumStable(out);
		DoubleObject ret = new DoubleObject(tmp.getValue(0, 0));
		sec.setVariable(output.getName(), ret);
	}
	
	/**
	 * 
	 */
	private static class RDDAggregateTernaryFunction 
		implements Function<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>, MatrixBlock>
	{
		private static final long serialVersionUID = 6410232464410434210L;
		
		private AggregateBinaryOperator _aggop = null;
		
		public RDDAggregateTernaryFunction( AggregateBinaryOperator aggop ) 
		{
			_aggop = aggop;		
		}
	
		@Override
		public MatrixBlock call(Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock> arg0)
			throws Exception 
		{
			//get inputs
			MatrixBlock in1 = arg0._1()._1();
			MatrixBlock in2 = arg0._1()._2();
			MatrixBlock in3 = arg0._2();
			
			//execute aggregate ternary operation
			ScalarObject ret = in1.aggregateTernaryOperations(in1, in2, in3, _aggop);
			
			//create output matrix block (w/ correction)
			MatrixBlock out = new MatrixBlock(2,1,false);
			out.setValue(0, 0, ret.getDoubleValue());
			return out;
		}
	}
	
	/**
	 * 
	 */
	private static class RDDAggregateTernaryFunction2 
		implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock>
	{
		private static final long serialVersionUID = -6615412819746331700L;
		
		private AggregateBinaryOperator _aggop = null;
		
		public RDDAggregateTernaryFunction2( AggregateBinaryOperator aggop ) 
		{
			_aggop = aggop;		
		}
	
		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0)
			throws Exception 
		{
			//get inputs
			MatrixBlock in1 = arg0._1();
			MatrixBlock in2 = arg0._2();
			
			//execute aggregate ternary operation
			ScalarObject ret = in1.aggregateTernaryOperations(in1, in2, null, _aggop);
			
			//create output matrix block (w/ correction)
			MatrixBlock out = new MatrixBlock(2,1,false);
			out.setValue(0, 0, ret.getDoubleValue());
			return out;
		}
	}
}
