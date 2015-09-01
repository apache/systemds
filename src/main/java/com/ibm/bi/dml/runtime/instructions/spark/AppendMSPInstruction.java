/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.utils.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

public class AppendMSPInstruction extends BinarySPInstruction
{
	
	private CPOperand _offset = null;
	
	public AppendMSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand offset, CPOperand out, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAppend;
			
		_offset = offset;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and offset = 4
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand offset = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
				
		if(!opcode.equalsIgnoreCase("mappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendMSPInstruction: " + str);
		else
			return new AppendMSPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, offset, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		// map-only append (rhs must be vector and fit in mapper mem)
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		int brlen = mc1.getRowsPerBlock();
		int bclen = mc1.getColsPerBlock();
		
		if(!mc1.dimsKnown() || !mc2.dimsKnown()) {
			throw new DMLRuntimeException("The dimensions unknown for inputs");
		}
		else if(mc1.getRows() != mc2.getRows()) {
			throw new DMLRuntimeException("The number of rows of inputs should match for append instruction");
		}
		else if(brlen != mc2.getRowsPerBlock() || bclen != mc2.getColsPerBlock()) {
			throw new DMLRuntimeException("The block sizes do not match for input matrices");
		}
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		Broadcast<PartitionedMatrixBlock> in2 = sec.getBroadcastForVariable( input2.getName() );
		long off = sec.getScalarInput( _offset.getName(), _offset.getValueType(), _offset.isLiteral()).getLongValue();
		
		//execute map-append operations (partitioning preserving if #in-blocks = #out-blocks)
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( preservesPartitioning(mc1, mc2) ){
			out = in1.mapPartitionsToPair(
					new MapSideAppendPartitionFunction(in2, off, bclen), true);
		}
		else {
			out = in1.flatMapToPair(
					new MapSideAppendFunction(in2, off, brlen, bclen));
		}
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputMatrixCharacteristics(sec);
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
	private boolean preservesPartitioning( MatrixCharacteristics mcIn1, MatrixCharacteristics mcIn2 )
	{
		long ncblksIn1 = (long)Math.ceil((double)mcIn1.getCols()/mcIn1.getColsPerBlock());
		long ncblksOut = (long)Math.ceil(((double)mcIn1.getCols()+mcIn2.getCols())/mcIn1.getColsPerBlock());
		
		//mappend is partitioning-preserving if in-block append (e.g., common case of colvector append)
		return (ncblksIn1 == ncblksOut);
	}
	
	/**
	 * 
	 */
	private static class MapSideAppendFunction implements  PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 2738541014432173450L;
		
		private Broadcast<PartitionedMatrixBlock> _pm = null;
		
		private long _offset; 
		private int _brlen; 
		private int _bclen;
		private long _lastBlockColIndex;
		
		public MapSideAppendFunction(Broadcast<PartitionedMatrixBlock> binput, long offset, int brlen, int bclen)  
		{
			_pm = binput;
			
			_offset = offset;
			_brlen = brlen;
			_bclen = bclen;
			
			//check for boundary block
			_lastBlockColIndex = (long)Math.ceil((double)_offset/bclen);
			
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			
			IndexedMatrixValue in1 = SparkUtils.toIndexedMatrixBlock(kv);
			
			//case 1: pass through of non-boundary blocks
			if( in1.getIndexes().getColumnIndex()!=_lastBlockColIndex ) {
				ret.add( kv );
			}
			//case 2: pass through full input block and rhs block 
			else if( in1.getValue().getNumColumns() == _bclen ) {				
				//output lhs block
				ret.add( kv );
				
				//output shallow copy of rhs block
				ret.add( new Tuple2<MatrixIndexes, MatrixBlock>(
						new MatrixIndexes(in1.getIndexes().getRowIndex(), in1.getIndexes().getColumnIndex()+1),
						_pm.getValue().getMatrixBlock((int)in1.getIndexes().getRowIndex(), 1)) );
			}
			//case 3: append operation on boundary block
			else 
			{
				MatrixBlock value_in2 = _pm.getValue().getMatrixBlock((int)in1.getIndexes().getRowIndex(), 1);
				
				//allocate space for the output value
				ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(2);
				IndexedMatrixValue first = new IndexedMatrixValue(new MatrixIndexes(in1.getIndexes()), new MatrixBlock());
				outlist.add(first);
				
				if(in1.getValue().getNumColumns()+value_in2.getNumColumns()>_bclen)
				{
					IndexedMatrixValue second=new IndexedMatrixValue(new MatrixIndexes(), new MatrixBlock());
					second.getIndexes().setIndexes(in1.getIndexes().getRowIndex(), in1.getIndexes().getColumnIndex()+1);
					outlist.add(second);
				}
	
				OperationsOnMatrixValues.performAppend(in1.getValue(), value_in2, outlist, _brlen, _bclen, true, 0);	
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

		private Broadcast<PartitionedMatrixBlock> _pm = null;
		private long _lastBlockColIndex;
		
		public MapSideAppendPartitionFunction(Broadcast<PartitionedMatrixBlock> binput, long offset, int bclen)  
		{
			_pm = binput;
			
			//check for boundary block
			_lastBlockColIndex = (long)Math.ceil((double)offset/bclen);			
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			
			//get the broadcast once
			PartitionedMatrixBlock pm = _pm.getValue();
			
			//process all blocks append operations
			while( arg0.hasNext() )
			{
				Tuple2<MatrixIndexes,MatrixBlock> tmp = arg0.next();
				MatrixIndexes ix = tmp._1();
				MatrixBlock in1 = tmp._2();
				
				//case 1: pass through of non-boundary blocks
				if( ix.getColumnIndex()!=_lastBlockColIndex ) {
					ret.add( tmp );
				}
				//case 3: append operation on boundary block
				else {
					MatrixBlock in2 = pm.getMatrixBlock((int)ix.getRowIndex(), 1);
					MatrixBlock out = in1.appendOperations(in2, new MatrixBlock());
					ret.add( new Tuple2<MatrixIndexes,MatrixBlock>(ix, out) );
				}	
			}
			
			return ret;
		}
	}
}
