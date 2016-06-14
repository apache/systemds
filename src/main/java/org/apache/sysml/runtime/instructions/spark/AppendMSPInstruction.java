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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;

public class AppendMSPInstruction extends BinarySPInstruction
{
	private CPOperand _offset = null;
	private boolean _cbind = true;
	
	public AppendMSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand offset, CPOperand out, boolean cbind, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAppend;			
		_offset = offset;
		_cbind = cbind;
	}
	
	public static AppendMSPInstruction parseInstruction ( String str ) 
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
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendMSPInstruction: " + str);
		
		return new AppendMSPInstruction(
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
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		int brlen = mc1.getRowsPerBlock();
		int bclen = mc1.getColsPerBlock();
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		PartitionedBroadcast in2 = sec.getBroadcastForVariable( input2.getName() );
		long off = sec.getScalarInput( _offset.getName(), _offset.getValueType(), _offset.isLiteral()).getLongValue();
		
		//execute map-append operations (partitioning preserving if #in-blocks = #out-blocks)
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( preservesPartitioning(mc1, mc2, _cbind) ) {
			out = in1.mapPartitionsToPair(
					new MapSideAppendPartitionFunction(in2, _cbind, off, brlen, bclen), true);
		}
		else {
			out = in1.flatMapToPair(
					new MapSideAppendFunction(in2, _cbind, off, brlen, bclen));
		}
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputMatrixCharacteristics(sec, _cbind);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageBroadcast(output.getName(), input2.getName());
	}
	
	/**
	 * 
	 * @param mcIn1
	 * @param mcIn2
	 * @return
	 */
	private boolean preservesPartitioning( MatrixCharacteristics mcIn1, MatrixCharacteristics mcIn2, boolean cbind )
	{
		long ncblksIn1 = cbind ?
				(long)Math.ceil((double)mcIn1.getCols()/mcIn1.getColsPerBlock()) : 
				(long)Math.ceil((double)mcIn1.getRows()/mcIn1.getRowsPerBlock());
		long ncblksOut = cbind ? 
				(long)Math.ceil(((double)mcIn1.getCols()+mcIn2.getCols())/mcIn1.getColsPerBlock()) : 
				(long)Math.ceil(((double)mcIn1.getRows()+mcIn2.getRows())/mcIn1.getRowsPerBlock());
		
		//mappend is partitioning-preserving if in-block append (e.g., common case of colvector append)
		return (ncblksIn1 == ncblksOut);
	}
	
	/**
	 * 
	 */
	private static class MapSideAppendFunction implements  PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 2738541014432173450L;
		
		private PartitionedBroadcast _pm = null;
		private boolean _cbind = true;
		private long _offset; 
		private int _brlen; 
		private int _bclen;
		private long _lastBlockColIndex;
		
		public MapSideAppendFunction(PartitionedBroadcast binput, boolean cbind, long offset, int brlen, int bclen)  
		{
			_pm = binput;
			_cbind = cbind;
			
			_offset = offset;
			_brlen = brlen;
			_bclen = bclen;
			
			//check for boundary block
			int blen = cbind ? bclen : brlen;
			_lastBlockColIndex = (long)Math.ceil((double)_offset/blen);			
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			
			IndexedMatrixValue in1 = SparkUtils.toIndexedMatrixBlock(kv);
			MatrixIndexes ix = in1.getIndexes();
			
			//case 1: pass through of non-boundary blocks
			if( (_cbind?ix.getColumnIndex():ix.getRowIndex())!=_lastBlockColIndex ) 
			{
				ret.add( kv );
			}
			//case 2: pass through full input block and rhs block 
			else if( _cbind && in1.getValue().getNumColumns() == _bclen 
					|| !_cbind && in1.getValue().getNumRows() == _brlen) 
			{				
				//output lhs block
				ret.add( kv );
				
				//output shallow copy of rhs block
				if( _cbind ) {
					ret.add( new Tuple2<MatrixIndexes, MatrixBlock>(
							new MatrixIndexes(ix.getRowIndex(), ix.getColumnIndex()+1),
							(MatrixBlock)_pm.getBlock((int)ix.getRowIndex(), 1)) );
				}
				else { //rbind
					ret.add( new Tuple2<MatrixIndexes, MatrixBlock>(
							new MatrixIndexes(ix.getRowIndex()+1, ix.getColumnIndex()),
							(MatrixBlock)_pm.getBlock(1, (int)ix.getColumnIndex())) );	
				}
			}
			//case 3: append operation on boundary block
			else 
			{
				//allocate space for the output value
				ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(2);
				IndexedMatrixValue first = new IndexedMatrixValue(new MatrixIndexes(ix), new MatrixBlock());
				outlist.add(first);
				
				MatrixBlock value_in2 = null;
				if( _cbind ) {
					value_in2 = (MatrixBlock)_pm.getBlock((int)ix.getRowIndex(), 1);
					if(in1.getValue().getNumColumns()+value_in2.getNumColumns()>_bclen) {
						IndexedMatrixValue second=new IndexedMatrixValue(new MatrixIndexes(), new MatrixBlock());
						second.getIndexes().setIndexes(ix.getRowIndex(), ix.getColumnIndex()+1);
						outlist.add(second);
					}
				}
				else { //rbind
					value_in2 = (MatrixBlock)_pm.getBlock(1, (int)ix.getColumnIndex());
					if(in1.getValue().getNumRows()+value_in2.getNumRows()>_brlen) {
						IndexedMatrixValue second=new IndexedMatrixValue(new MatrixIndexes(), new MatrixBlock());
						second.getIndexes().setIndexes(ix.getRowIndex()+1, ix.getColumnIndex());
						outlist.add(second);
					}
				}
	
				OperationsOnMatrixValues.performAppend(in1.getValue(), value_in2, outlist, _brlen, _bclen, _cbind, true, 0);	
				ret.addAll(SparkUtils.fromIndexedMatrixBlock(outlist));
			}
			
			return ret;
		}
	}
	
	/**
	 * 
	 */
	private static class MapSideAppendPartitionFunction implements  PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 5767240739761027220L;

		private PartitionedBroadcast _pm = null;
		private boolean _cbind = true;
		private long _lastBlockColIndex = -1;
		
		public MapSideAppendPartitionFunction(PartitionedBroadcast binput, boolean cbind, long offset, int brlen, int bclen)  
		{
			_pm = binput;
			_cbind = cbind;
			
			//check for boundary block
			int blen = cbind ? bclen : brlen;
			_lastBlockColIndex = (long)Math.ceil((double)offset/blen);			
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new MapAppendPartitionIterator(arg0);
		}
		
		/**
		 * Lazy mappend iterator to prevent materialization of entire partition output in-memory.
		 * The implementation via mapPartitions is required to preserve partitioning information,
		 * which is important for performance. 
		 */
		private class MapAppendPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public MapAppendPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg)
				throws Exception
			{
				MatrixIndexes ix = arg._1();
				MatrixBlock in1 = arg._2();
				
				//case 1: pass through of non-boundary blocks
				if( (_cbind?ix.getColumnIndex():ix.getRowIndex()) != _lastBlockColIndex ) {
					return arg;
				}
				//case 3: append operation on boundary block
				else {
					int rowix = _cbind ? (int)ix.getRowIndex() : 1;
					int colix = _cbind ? 1 : (int)ix.getColumnIndex();					
					MatrixBlock in2 = (MatrixBlock)_pm.getBlock(rowix, colix);
					MatrixBlock out = in1.appendOperations(in2, new MatrixBlock(), _cbind);
					return new Tuple2<MatrixIndexes,MatrixBlock>(ix, out);
				}	
			}			
		}
	}
}
