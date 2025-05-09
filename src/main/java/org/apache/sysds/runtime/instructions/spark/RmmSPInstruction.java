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
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.data.TripleIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import scala.Tuple2;

import java.util.Iterator;
import java.util.LinkedList;

public class RmmSPInstruction extends AggregateBinarySPInstruction {

	private RmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(SPType.RMM, op, in1, in2, out, opcode, istr);
	}

	public static RmmSPInstruction parseInstruction( String str ) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( Opcodes.RMM.toString().equals(opcode) ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);

			return new RmmSPInstruction(null, in1, in2, out, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("RmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input rdds
		DataCharacteristics mc1 = sec.getDataCharacteristics( input1.getName() );
		DataCharacteristics mc2 = sec.getDataCharacteristics( input2.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() );
		DataCharacteristics mcOut = updateBinaryMMOutputDataCharacteristics(sec, true);
		
		//execute Spark RMM instruction
		//step 1: prepare join keys (w/ shallow replication), i/j/k
		JavaPairRDD<TripleIndexes,MatrixBlock> tmp1 = in1.flatMapToPair(
			new RmmReplicateFunction(mc2.getCols(), mc2.getBlocksize(), true)); 
		JavaPairRDD<TripleIndexes,MatrixBlock> tmp2 = in2.flatMapToPair(
			new RmmReplicateFunction(mc1.getRows(), mc1.getBlocksize(), false));
		
		//step 2: join prepared datasets, multiply, and aggregate
		int numPartJoin = Math.max(getNumJoinPartitions(mc1, mc2),
			SparkExecutionContext.getDefaultParallelism(true));
		int numPartOut = SparkUtils.getNumPreferredPartitions(mcOut);
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = tmp1
			.join( tmp2, numPartJoin )               //join by result block 
		    .mapToPair( new RmmMultiplyFunction() ); //do matrix multiplication
		out = RDDAggregateUtils.sumByKeyStable(out,  //aggregation per result block
			numPartOut, false); 
		
		//put output block into symbol table (no lineage because single block)
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}
	
	private static int getNumJoinPartitions(DataCharacteristics mc1, DataCharacteristics mc2) {
		if( !mc1.dimsKnown() || !mc2.dimsKnown() )
			SparkExecutionContext.getDefaultParallelism(true);
		//compute data size of replicated inputs
		double hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
		double matrix1PSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(mc1)
			* ((long) Math.ceil((double)mc2.getCols()/mc2.getBlocksize()));
		double matrix2PSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(mc2)
			* ((long) Math.ceil((double)mc1.getRows()/mc1.getBlocksize()));
		return (int) Math.max(Math.ceil((matrix1PSize+matrix2PSize)/hdfsBlockSize), 1);
	}

	private static class RmmReplicateFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, TripleIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 3577072668341033932L;
		
		private long _len = -1;
		private long _blen = -1;
		private boolean _left = false;
		
		public RmmReplicateFunction(long len, long blen, boolean left)
		{
			_len = len;
			_blen = blen;
			_left = left;
		}
		
		@Override
		public Iterator<Tuple2<TripleIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			LinkedList<Tuple2<TripleIndexes, MatrixBlock>> ret = new LinkedList<>();
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			long numBlocks = (long) Math.ceil((double)_len/_blen); 
			
			if( _left ) //LHS MATRIX
			{
				//replicate wrt # column blocks in RHS
				long i = ixIn.getRowIndex();
				long k = ixIn.getColumnIndex();
				for( long j=1; j<=numBlocks; j++ ) {
					TripleIndexes tmptix = new TripleIndexes(i, j, k);
					ret.add( new Tuple2<>(tmptix, blkIn) );
				}
			} 
			else // RHS MATRIX
			{
				//replicate wrt # row blocks in LHS
				long k = ixIn.getRowIndex();
				long j = ixIn.getColumnIndex();
				for( long i=1; i<=numBlocks; i++ ) {
					TripleIndexes tmptix = new TripleIndexes(i, j, k);
					ret.add( new Tuple2<>(tmptix, blkIn) );
				}
			}
			
			//output list of new tuples
			return ret.iterator();
		}
	}

	private static class RmmMultiplyFunction implements PairFunction<Tuple2<TripleIndexes, Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -5772410117511730911L;
		
		private AggregateBinaryOperator _op = null;
		
		public RmmMultiplyFunction()
		{
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<TripleIndexes, Tuple2<MatrixBlock,MatrixBlock>> arg0 ) 
			throws Exception 
		{
			//get input blocks per
			TripleIndexes ixIn = arg0._1(); //i,j,k
			MatrixIndexes ixOut = new MatrixIndexes(ixIn.getFirstIndex(), ixIn.getSecondIndex()); //i,j
			MatrixBlock blkIn1 = arg0._2()._1();
			MatrixBlock blkIn2 = arg0._2()._2();
			
			//core block matrix multiplication 
			MatrixBlock blkOut = OperationsOnMatrixValues.matMult(blkIn1, blkIn2, new MatrixBlock(), _op);
			
			//output new tuple
			return new Tuple2<>(ixOut, blkOut);
		}
	}
}
