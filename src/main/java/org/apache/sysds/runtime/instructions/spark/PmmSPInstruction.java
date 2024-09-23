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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.MapMult.CacheType;
import org.apache.sysds.lops.PMMJ;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

public class PmmSPInstruction extends BinarySPInstruction {
	private CacheType _type = null;
	private CPOperand _nrow = null;

	private PmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, CPOperand nrow, CacheType type,
			String opcode, String istr) {
		super(SPType.PMM, op, in1, in2, out, opcode, istr);
		_type = type;
		_nrow = nrow;
	}

	public static PmmSPInstruction parseInstruction( String str ) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = InstructionUtils.getOpCode(str);
		if ( opcode.equalsIgnoreCase(PMMJ.OPCODE)) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand nrow = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			CacheType type = CacheType.valueOf(parts[5]);
			AggregateBinaryOperator aggbin = InstructionUtils.getMatMultOperator(1);
			return new PmmSPInstruction(aggbin, in1, in2, out, nrow, type, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("PmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		String rddVar = (_type==CacheType.LEFT) ? input2.getName() : input1.getName();
		String bcastVar = (_type==CacheType.LEFT) ? input1.getName() : input2.getName();
		DataCharacteristics mc = sec.getDataCharacteristics(output.getName());
		long rlen = sec.getScalarInput(_nrow).getLongValue();
		
		//get inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( rddVar );
		PartitionedBroadcast<MatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar ); 
		
		//execute pmm instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1
				.flatMapToPair( new RDDPMMFunction(_type, in2, rlen, mc.getBlocksize()) );
		out = RDDAggregateUtils.sumByKeyStable(out, false);
		
		//put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), rddVar);
		sec.addLineageBroadcast(output.getName(), bcastVar);
		
		//update output statistics if not inferred
		updateBinaryMMOutputDataCharacteristics(sec, false);
	}

	public CacheType getCacheType() { return _type; }

	private static class RDDPMMFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -1696560050436469140L;
		
		private PartitionedBroadcast<MatrixBlock> _pmV = null;
		private long _rlen = -1;
		private int _blen = -1;
		
		public RDDPMMFunction( CacheType type, PartitionedBroadcast<MatrixBlock> binput, long rlen, int blen ) {
			_blen = blen;
			_rlen = rlen;
			_pmV = binput;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) {
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock mb2 = arg0._2();
			
			//get the right hand side matrix
			MatrixBlock mb1 = _pmV.getBlock((int)ixIn.getRowIndex(), 1);
			
			//compute target block indexes
			long minPos = UtilFunctions.toLong( mb1.minNonZero() );
			long maxPos = UtilFunctions.toLong( mb1.max() );
			long rowIX1 = (minPos-1)/_blen+1;
			long rowIX2 = (maxPos-1)/_blen+1;
			boolean multipleOuts = (rowIX1 != rowIX2);
			
			if( minPos >= 1 ) //at least one row selected
			{
				//output sparsity estimate
				double spmb1 = OptimizerUtils.getSparsity(mb1.getNumRows(), 1, mb1.getNonZeros());
				long estnnz = (long) (spmb1 * mb2.getNonZeros());
				boolean sparse = MatrixBlock.evalSparseFormatInMemory(_blen, mb2.getNumColumns(), estnnz);
				
				//compute and allocate output blocks
				MatrixBlock out1 = new MatrixBlock();
				MatrixBlock out2 = multipleOuts ? new MatrixBlock() : null;
				out1.reset(_blen, mb2.getNumColumns(), sparse);
				if( out2 != null )
					out2.reset(UtilFunctions.computeBlockSize(_rlen, rowIX2, _blen), mb2.getNumColumns(), sparse);
				
				//compute core matrix permutation (assumes that out1 has default blocksize, 
				//hence we do a meta data correction afterwards)
				mb1.permutationMatrixMultOperations(mb2, out1, out2);
				out1.setNumRows(UtilFunctions.computeBlockSize(_rlen, rowIX1, _blen));
				ret.add(new Tuple2<>(new MatrixIndexes(rowIX1, ixIn.getColumnIndex()), out1));
				if( out2 != null )
					ret.add(new Tuple2<>(new MatrixIndexes(rowIX2, ixIn.getColumnIndex()), out2));
			}
			
			return ret.iterator();
		}
	}
}
