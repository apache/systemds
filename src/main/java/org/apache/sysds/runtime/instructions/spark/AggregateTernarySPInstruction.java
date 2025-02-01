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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.functions.AggregateDropCorrectionFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import scala.Tuple2;

public class AggregateTernarySPInstruction extends ComputationSPInstruction {

	private AggregateTernarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr) {
		super(SPType.AggregateTernary, op, in1, in2, in3, out, opcode, istr);
	}

	public static AggregateTernarySPInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if ( opcode.equalsIgnoreCase(Opcodes.TAKPM.toString()) || opcode.equalsIgnoreCase(Opcodes.TACKPM.toString()) ) {
			InstructionUtils.checkNumFields( parts, 4 );
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			AggregateTernaryOperator op = InstructionUtils.parseAggregateTernaryOperator(opcode);
			return new AggregateTernarySPInstruction(op, in1, in2, in3, out, opcode, str);
		}
		throw new DMLRuntimeException("AggregateTernaryInstruction.parseInstruction():: Unknown opcode " + opcode);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get inputs
		DataCharacteristics mcIn = sec.getDataCharacteristics( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = input3.isLiteral() ? null : //matrix or literal 1
													 sec.getBinaryMatrixBlockRDDHandleForVariable( input3.getName() );
		
		//execute aggregate ternary operation
		AggregateTernaryOperator aggop = (AggregateTernaryOperator) _optr;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( in3 != null ) { //3 inputs
			out = in1.join( in2 ).join( in3 )
				     .mapToPair(new RDDAggregateTernaryFunction(aggop));
		}
		else { //2 inputs (third is literal 1)
			out = in1.join( in2 )
					 .mapToPair(new RDDAggregateTernaryFunction2(aggop));
		}
		
		//aggregate partial results
		if( aggop.indexFn instanceof ReduceAll ) //tak+*
		{
			//aggregate and create output (no lineage because scalar)
			MatrixBlock tmp = RDDAggregateUtils.sumStable(out.values());
			DoubleObject ret = new DoubleObject(tmp.get(0, 0));
			sec.setVariable(output.getName(), ret);	
		}
		else if( mcIn.dimsKnown() && mcIn.getCols()<=mcIn.getBlocksize() ) //tack+* single block
		{
			//single block aggregation and drop correction
			MatrixBlock ret = RDDAggregateUtils.aggStable(out, aggop.aggOp);
			ret.dropLastRowsOrColumns(aggop.aggOp.correction);
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), ret);
		}
		else //tack+* multi block
		{
			//multi-block aggregation and drop correction
			out = RDDAggregateUtils.aggByKeyStable(out, aggop.aggOp, false);
			out = out.mapValues( new AggregateDropCorrectionFunction(aggop.aggOp) );

			//put output RDD handle into symbol table
			updateUnaryAggOutputDataCharacteristics(sec, aggop.indexFn);
			sec.setRDDHandleForVariable(output.getName(), out);	
			sec.addLineageRDD(output.getName(), input1.getName());
			sec.addLineageRDD(output.getName(), input2.getName());
			if( in3 != null )
				sec.addLineageRDD(output.getName(), input3.getName());
		}
	}

	private static class RDDAggregateTernaryFunction 
		implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 6410232464410434210L;
		
		private final AggregateTernaryOperator _aggop;
		
		public RDDAggregateTernaryFunction( AggregateTernaryOperator aggop ) {
			_aggop = aggop;
		}
	
		@Override
		public Tuple2<MatrixIndexes,MatrixBlock> call(Tuple2<MatrixIndexes,Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>> arg0)
			throws Exception 
		{
			//get inputs
			MatrixIndexes ix = arg0._1();
			MatrixBlock in1 = arg0._2()._1()._1();
			MatrixBlock in2 = arg0._2()._1()._2();
			MatrixBlock in3 = arg0._2()._2();
			
			//execute aggregate ternary operation
			return new Tuple2<>(new MatrixIndexes(1, ix.getColumnIndex()),
				MatrixBlock.aggregateTernaryOperations(in1, in2, in3, new MatrixBlock(), _aggop, false));
		}
	}

	private static class RDDAggregateTernaryFunction2 
		implements PairFunction<Tuple2<MatrixIndexes,Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -6615412819746331700L;
		
		private final AggregateTernaryOperator _aggop;
		
		public RDDAggregateTernaryFunction2( AggregateTernaryOperator aggop ) {
			_aggop = aggop;		
		}
	
		@Override
		public Tuple2<MatrixIndexes,MatrixBlock> call(Tuple2<MatrixIndexes,Tuple2<MatrixBlock, MatrixBlock>> arg0)
			throws Exception 
		{
			//get inputs
			MatrixIndexes ix = arg0._1();
			MatrixBlock in1 = arg0._2()._1();
			MatrixBlock in2 = arg0._2()._2();
			
			//execute aggregate ternary operation
			return new Tuple2<>(new MatrixIndexes(1, ix.getColumnIndex()),
				MatrixBlock.aggregateTernaryOperations(in1, in2, null, new MatrixBlock(), _aggop, false));
		}
	}
}
