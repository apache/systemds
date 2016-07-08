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

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;

public class FrameAppendMSPInstruction extends AppendMSPInstruction
{
	public FrameAppendMSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand offset, CPOperand out, boolean cbind, String opcode, String istr)
	{
		super(op, in1, in2, offset, out, cbind, opcode, istr);
	}
	
	public static FrameAppendMSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 5);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand offset = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		boolean cbind = Boolean.parseBoolean(parts[5]);
		
		if(!opcode.equalsIgnoreCase("mappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a FrameAppendMSPInstruction: " + str);
		
		return new FrameAppendMSPInstruction(
				new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
				in1, in2, offset, out, cbind, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		// map-only append (rhs must be vector and fit in mapper mem)
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		checkBinaryAppendInputCharacteristics(sec, _cbind, false, false);
		
		JavaPairRDD<Long,FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable( input1.getName() );
		PartitionedBroadcast<FrameBlock> in2 = sec.getBroadcastForFrameVariable( input2.getName() );
		long off = sec.getScalarInput( _offset.getName(), _offset.getValueType(), _offset.isLiteral()).getLongValue();
		
		//execute map-append operations (partitioning preserving if #in-blocks = #out-blocks)
		JavaPairRDD<Long,FrameBlock> out = null;
		if( preservesPartitioning(_cbind) ) {
			out = in1.mapPartitionsToPair(
					new MapSideAppendPartitionFunction(in2, _cbind, off), true);
		}
		else {
			out = in1.flatMapToPair(
					new MapSideAppendFunction(in2, _cbind, off));
		}
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputMatrixCharacteristics(sec, _cbind);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageBroadcast(output.getName(), input2.getName());
	}
	
	/** 
	 * 
	 * @param cbind
	 * @return
	 */
	private boolean preservesPartitioning( boolean cbind )
	{
		//Partitions for input1 will be preserved in case of cbind, 
		// where as in case of rbind partitions for input2 (added in output) will not be preserved.
		return cbind;
	}
	
	/**
	 * 
	 */
	private static class MapSideAppendFunction implements  PairFlatMapFunction<Tuple2<Long,FrameBlock>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = -4536950734144427203L;

		private PartitionedBroadcast<FrameBlock> _pm = null;
		private boolean _cbind = true;
		private long _offset; 
		
		public MapSideAppendFunction(PartitionedBroadcast<FrameBlock> binput, boolean cbind, long offset)  
		{
			_pm = binput;
			_cbind = cbind;
			_offset = offset;
		}
		
		@Override
		public Iterable<Tuple2<Long, FrameBlock>> call(Tuple2<Long, FrameBlock> kv) 
			throws Exception 
		{
			ArrayList<Tuple2<Long, FrameBlock>> ret = new ArrayList<Tuple2<Long, FrameBlock>>();
			
			FrameBlock in1 = kv._2();
			Long ix = kv._1();
			
			//case 1: pass through of non-boundary blocks
			if(!_cbind && ix < _offset) 
			{
				ret.add( kv ); 
			}
			//case 2: change key of rhs blocks in boundary case 
			else if(!_cbind && ix >= _offset)
			{				
				//output lhs block
				ret.add( kv );

				//output rhs block with new key
				ret.add( new Tuple2<Long, FrameBlock>(
						ix+kv._2().getNumRows(),
						_pm.getBlock(1, 1)) );	
			}
			// will not hit this case
			else 
			{
				FrameBlock right = _pm.getBlock((ix.intValue()-1)/OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE+1, 1);
				FrameBlock out = in1.appendOperations(right, new FrameBlock(), _cbind);
				ret.add(new Tuple2<Long, FrameBlock>(ix, out));
			}
			
			return ret;
		}
	}
	
	/**
	 * 
	 */
	private static class MapSideAppendPartitionFunction implements  PairFlatMapFunction<Iterator<Tuple2<Long,FrameBlock>>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = -3997051891171313830L;

		private PartitionedBroadcast<FrameBlock> _pm = null;
		private boolean _cbind = true;
		private long _offset;
		
		public MapSideAppendPartitionFunction(PartitionedBroadcast<FrameBlock> binput, boolean cbind, long offset)  
		{
			_pm = binput;
			_cbind = cbind;
			_offset = offset;
		}

		@Override
		public Iterable<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Long, FrameBlock>> arg0)
			throws Exception 
		{
			return new MapAppendPartitionIterator(arg0);
		}
		
		/**
		 * Lazy mappend iterator to prevent materialization of entire partition output in-memory.
		 * The implementation via mapPartitions is required to preserve partitioning information,
		 * which is important for performance. 
		 */
		private class MapAppendPartitionIterator extends LazyIterableIterator<Tuple2<Long, FrameBlock>>
		{
			public MapAppendPartitionIterator(Iterator<Tuple2<Long, FrameBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<Long, FrameBlock> computeNext(Tuple2<Long, FrameBlock> arg)
				throws Exception
			{
				Long ix = arg._1();
				FrameBlock in1 = arg._2();
			
				//Non existing case
				if( (!_cbind) && (_offset >= ix && ix < _offset+in1.getNumRows())) {	
					return arg;
				}
				//case : append operation on boundary block
				else {
					int rowix = _cbind ? (ix.intValue()-1)/OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE+1 : 1;
					int colix = 1;
					
					FrameBlock in2 = _pm.getBlock(rowix, colix);
					FrameBlock out = in1.appendOperations(in2, new FrameBlock(), _cbind);
					return new Tuple2<Long,FrameBlock>(ix, out);
				}	
			}			
		}
	}
}
