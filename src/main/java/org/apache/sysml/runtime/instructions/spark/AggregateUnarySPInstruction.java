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
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.hops.AggBinaryOp.SparkAggType;
import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.functions.AggregateDropCorrectionFunction;
import org.apache.sysml.runtime.instructions.spark.functions.FilterDiagBlocksFunction;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * 
 */
public class AggregateUnarySPInstruction extends UnarySPInstruction
{
	
	private SparkAggType _aggtype = null;
	private AggregateOperator _aop = null;
	
	public AggregateUnarySPInstruction(AggregateUnaryOperator auop, AggregateOperator aop, CPOperand in, CPOperand out, SparkAggType aggtype, String opcode, String istr){
		super(auop, in, out, opcode, istr);
		_aggtype = aggtype;
		_aop = aop;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static AggregateUnarySPInstruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		SparkAggType aggtype = SparkAggType.valueOf(parts[3]);
		
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(opcode);
		CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(opcode);
		String corrExists = (corrLoc != CorrectionLocationType.NONE) ? "true" : "false";
		
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrExists, corrLoc.toString());
		return new AggregateUnarySPInstruction(aggun, aop, in1, out, aggtype, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in;
		
		//filter input blocks for trace
		if( getOpcode().equalsIgnoreCase("uaktrace") )
			out = out.filter(new FilterDiagBlocksFunction());
		
		//execute unary aggregate operation
		AggregateUnaryOperator auop = (AggregateUnaryOperator)_optr;
		AggregateOperator aggop = _aop;
		if( _aggtype == SparkAggType.NONE ) {
			//in case of no block aggregation, we always drop the correction as well as
			//use a partitioning-preserving mapvalues 
			out = out.mapValues(new RDDUAggValueFunction(auop, mc.getRowsPerBlock(), mc.getColsPerBlock()));
		}
		else {
			//in case of single/multi-block aggregation, we always keep the correction
			out = out.mapToPair(new RDDUAggFunction(auop, mc.getRowsPerBlock(), mc.getColsPerBlock()));			
		}
		
		
		//perform aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			MatrixBlock out2 = RDDAggregateUtils.aggStable(out, aggop);
			
			//drop correction after aggregation
			out2.dropLastRowsOrColums(aggop.correctionLocation);
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);
		}
		else //MULTI_BLOCK or NONE
		{
			if( _aggtype == SparkAggType.MULTI_BLOCK ) {
				out = RDDAggregateUtils.aggByKeyStable(out, aggop);
	
				//drop correction after aggregation if required (aggbykey creates 
				//partitioning, drop correction via partitioning-preserving mapvalues)
				if( auop.aggOp.correctionExists )
					out = out.mapValues( new AggregateDropCorrectionFunction(aggop) );
			}
			
			//put output RDD handle into symbol table
			updateUnaryAggOutputMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);	
			sec.addLineageRDD(output.getName(), input1.getName());
		}		
	}
	
	/**
	 * 
	 * @param sec
	 * @param auop
	 * @throws DMLRuntimeException
	 */
	protected void updateUnaryAggOutputMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		AggregateUnaryOperator auop = (AggregateUnaryOperator)_optr;
		
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			if(!mc1.dimsKnown()) {
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + mc1.toString() + " " + mcOut.toString());
			}
			else {
				//infer statistics from input based on operator
				if( auop.indexFn instanceof ReduceAll )
					mcOut.set(1, 1, mc1.getRowsPerBlock(), mc1.getColsPerBlock());
				else if (auop.indexFn instanceof ReduceCol)
					mcOut.set(mc1.getRows(), 1, mc1.getRowsPerBlock(), mc1.getColsPerBlock());
				else if (auop.indexFn instanceof ReduceRow)
					mcOut.set(1, mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
		}
	}

	/**
	 * 
	 */
	private static class RDDUAggFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 2672082409287856038L;
		
		private AggregateUnaryOperator _op = null;
		private int _brlen = -1;
		private int _bclen = -1;
		
		public RDDUAggFunction( AggregateUnaryOperator op, int brlen, int bclen )
		{
			_op = op;
			_brlen = brlen;
			_bclen = bclen;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			//unary aggregate operation (always keep the correction)
			OperationsOnMatrixValues.performAggregateUnary( ixIn, blkIn, 
					  ixOut, blkOut, _op, _brlen, _bclen);
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
	
	/**
	 * 
	 */
	private static class RDDUAggValueFunction implements Function<MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 5352374590399929673L;
		
		private AggregateUnaryOperator _op = null;
		private int _brlen = -1;
		private int _bclen = -1;
		private MatrixIndexes _ix = null;
		
		public RDDUAggValueFunction( AggregateUnaryOperator op, int brlen, int bclen )
		{
			_op = op;
			_brlen = brlen;
			_bclen = bclen;
			
			_ix = new MatrixIndexes(1,1);
		}
		
		@Override
		public MatrixBlock call( MatrixBlock arg0 ) 
			throws Exception 
		{
			MatrixBlock blkOut = new MatrixBlock();
			
			//unary aggregate operation
			arg0.aggregateUnaryOperations(_op, blkOut, _brlen, _bclen, _ix);
			
			//always drop correction since no aggregation
			blkOut.dropLastRowsOrColums(_op.aggOp.correctionLocation);
			
			//output new tuple
			return blkOut;
		}
	}
}
