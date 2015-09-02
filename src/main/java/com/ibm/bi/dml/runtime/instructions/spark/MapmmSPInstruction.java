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
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.hops.AggBinaryOp.SparkAggType;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.MapMult.CacheType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.LazyIterableIterator;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * TODO: we need to reason about multiple broadcast variables for chains of mapmults (sum of operations until cleanup) 
 * 
 */
public class MapmmSPInstruction extends BinarySPInstruction 
{
	
	private CacheType _type = null;
	private boolean _outputEmpty = true;
	private SparkAggType _aggtype;
	
	public MapmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, CacheType type, 
			                    boolean outputEmpty, SparkAggType aggtype, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAPMM;
		
		_type = type;
		_outputEmpty = outputEmpty;
		_aggtype = aggtype;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MapmmSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);

		if ( opcode.equalsIgnoreCase(MapMult.OPCODE)) {
			String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			CacheType type = CacheType.valueOf(parts[4]);
			boolean outputEmpty = Boolean.parseBoolean(parts[5]);
			SparkAggType aggtype = SparkAggType.valueOf(parts[6]);
			
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new MapmmSPInstruction(aggbin, in1, in2, out, type, outputEmpty, aggtype, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("MapmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		String rddVar = (_type==CacheType.LEFT) ? input2.getName() : input1.getName();
		String bcastVar = (_type==CacheType.LEFT) ? input1.getName() : input2.getName();
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		MatrixCharacteristics mcRdd = sec.getMatrixCharacteristics(rddVar);
		MatrixCharacteristics mcBc = sec.getMatrixCharacteristics(bcastVar);
		
		//get inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
		Broadcast<PartitionedMatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar ); 
				
		//empty input block filter
		if( !_outputEmpty )
			in1 = in1.filter(new FilterNonEmptyBlocksFunction());
			
		//execute mapmult instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( requiresFlatMapFunction(_type, mcBc) ) 
			out = in1.flatMapToPair( new RDDFlatMapMMFunction(_type, in2, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()) );
		else if( preservesPartitioning(mcRdd, _type) )
			out = in1.mapPartitionsToPair(new RDDMapMMPartitionFunction(_type, in2, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()), true);
		else
			out = in1.mapToPair( new RDDMapMMFunction(_type, in2, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()) );
		
		//empty output block filter
		if( !_outputEmpty )
			in1 = in1.filter(new FilterNonEmptyBlocksFunction());
		
		//perform aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			MatrixBlock out2 = RDDAggregateUtils.sumStable(out);
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);
		}
		else //MULTI_BLOCK or NONE
		{
			if( _aggtype == SparkAggType.MULTI_BLOCK )
				out = RDDAggregateUtils.sumByKeyStable(out);
		
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			sec.addLineageBroadcast(output.getName(), bcastVar);
			
			//update output statistics if not inferred
			updateBinaryMMOutputMatrixCharacteristics(sec, true);
		}
	}
	
	/**
	 * 
	 * @param mcIn
	 * @param type
	 * @return
	 */
	private boolean preservesPartitioning( MatrixCharacteristics mcIn, CacheType type )
	{
		if( type == CacheType.LEFT )
			return mcIn.dimsKnown() && mcIn.getRows() <= mcIn.getRowsPerBlock();
		else // RIGHT
			return mcIn.dimsKnown() && mcIn.getCols() <= mcIn.getColsPerBlock();
	}
	
	/**
	 * 
	 * @param type
	 * @param mcBc
	 * @return
	 */
	private static boolean requiresFlatMapFunction( CacheType type, MatrixCharacteristics mcBc)
	{
		return    (type == CacheType.LEFT && mcBc.getRows() > mcBc.getRowsPerBlock())
			   || (type == CacheType.RIGHT && mcBc.getCols() > mcBc.getColsPerBlock());
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDMapMMFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private CacheType _type = null;
		private AggregateBinaryOperator _op = null;
		private Broadcast<PartitionedMatrixBlock> _pbc = null;
		
		public RDDMapMMFunction( CacheType type, Broadcast<PartitionedMatrixBlock> binput, int brlen, int bclen )
		{
			_type = type;
			
			//partition vector for fast in memory lookup
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			PartitionedMatrixBlock pm = _pbc.value();
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			if( _type == CacheType.LEFT )
			{
				//get the right hand side matrix
				MatrixBlock left = pm.getMatrixBlock(1, (int)ixIn.getRowIndex());
				
				//execute matrix-vector mult
				OperationsOnMatrixValues.performAggregateBinary( 
						new MatrixIndexes(1,ixIn.getRowIndex()), left, ixIn, blkIn, ixOut, blkOut, _op);						
			}
			else //if( _type == CacheType.RIGHT )
			{
				//get the right hand side matrix
				MatrixBlock right = pm.getMatrixBlock((int)ixIn.getColumnIndex(), 1);
				
				//execute matrix-vector mult
				OperationsOnMatrixValues.performAggregateBinary(
						ixIn, blkIn, new MatrixIndexes(ixIn.getColumnIndex(),1), right, ixOut, blkOut, _op);					
			}
			
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}

	/**
	 * 
	 */
	private static class RDDMapMMPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1886318890063064287L;
	
		private CacheType _type = null;
		private AggregateBinaryOperator _op = null;
		private Broadcast<PartitionedMatrixBlock> _pbc = null;
		
		public RDDMapMMPartitionFunction( CacheType type, Broadcast<PartitionedMatrixBlock> binput, int brlen, int bclen )
		{
			_type = type;
			
			//partition vector for fast in memory lookup
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new MapMMPartitionIterator(arg0);
		}
		
		/**
		 * Lazy mapmm iterator to prevent materialization of entire partition output in-memory.
		 * The implementation via mapPartitions is required to preserve partitioning information,
		 * which is important for performance. 
		 */
		private class MapMMPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public MapMMPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg)
				throws Exception
			{
				MatrixIndexes ixIn = arg._1();
				MatrixBlock blkIn = arg._2();
				MatrixBlock blkOut = new MatrixBlock();
		
				if( _type == CacheType.LEFT )
				{
					//get the right hand side matrix
					MatrixBlock left = _pbc.value().getMatrixBlock(1, (int)ixIn.getRowIndex());
					
					//execute index preserving matrix multiplication
					left.aggregateBinaryOperations(left, blkIn, blkOut, _op);						
				}
				else //if( _type == CacheType.RIGHT )
				{
					//get the right hand side matrix
					MatrixBlock right = _pbc.value().getMatrixBlock((int)ixIn.getColumnIndex(), 1);

					//execute index preserving matrix multiplication
					blkIn.aggregateBinaryOperations(blkIn, right, blkOut, _op);					
				}
			
				return new Tuple2<MatrixIndexes,MatrixBlock>(ixIn, blkOut);
			}			
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDFlatMapMMFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -6076256569118957281L;
		
		private CacheType _type = null;
		private AggregateBinaryOperator _op = null;
		private Broadcast<PartitionedMatrixBlock> _pbc = null;
		
		public RDDFlatMapMMFunction( CacheType type, Broadcast<PartitionedMatrixBlock> binput, int brlen, int bclen )
		{
			_type = type;
			
			//partition vector for fast in memory lookup
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			PartitionedMatrixBlock pm = _pbc.value();
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			if( _type == CacheType.LEFT )
			{
				//for all matching left-hand-side blocks
				for( int i=1; i<=pm.getNumRowBlocks(); i++ ) 
				{
					MatrixBlock left = pm.getMatrixBlock(i, (int)ixIn.getRowIndex());
					MatrixIndexes ixOut = new MatrixIndexes();
					MatrixBlock blkOut = new MatrixBlock();
					
					//execute matrix-vector mult
					OperationsOnMatrixValues.performAggregateBinary( 
							new MatrixIndexes(i,ixIn.getRowIndex()), left, ixIn, blkIn, ixOut, blkOut, _op);	
					
					ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut));
				}
			}
			else //if( _type == CacheType.RIGHT )
			{
				//for all matching right-hand-side blocks
				for( int j=1; j<=pm.getNumColumnBlocks(); j++ ) 
				{
					//get the right hand side matrix
					MatrixBlock right = pm.getMatrixBlock((int)ixIn.getColumnIndex(), j);
					MatrixIndexes ixOut = new MatrixIndexes();
					MatrixBlock blkOut = new MatrixBlock();
					
					//execute matrix-vector mult
					OperationsOnMatrixValues.performAggregateBinary(
							ixIn, blkIn, new MatrixIndexes(ixIn.getColumnIndex(),j), right, ixOut, blkOut, _op);					
				
					ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut));
				}
			}
			
			return ret;
		}
	}
}
