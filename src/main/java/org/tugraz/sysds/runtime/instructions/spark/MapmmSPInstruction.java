/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark;


import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.tugraz.sysds.hops.AggBinaryOp.SparkAggType;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.MapMult;
import org.tugraz.sysds.lops.MapMult.CacheType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.functionobjects.Multiply;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.tugraz.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.tugraz.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import scala.Tuple2;

import java.util.Iterator;
import java.util.stream.IntStream;

public class MapmmSPInstruction extends BinarySPInstruction {
	private CacheType _type = null;
	private boolean _outputEmpty = true;
	private SparkAggType _aggtype;

	private MapmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, CacheType type,
			boolean outputEmpty, SparkAggType aggtype, String opcode, String istr) {
		super(SPType.MAPMM, op, in1, in2, out, opcode, istr);
		_type = type;
		_outputEmpty = outputEmpty;
		_aggtype = aggtype;
	}

	public static MapmmSPInstruction parseInstruction( String str ) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(!opcode.equalsIgnoreCase(MapMult.OPCODE)) 
			throw new DMLRuntimeException("MapmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
			
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
	
	@Override
	public void processInstruction(ExecutionContext ec)  {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		CacheType type = _type;
		String rddVar = type.isRight() ? input1.getName() : input2.getName();
		String bcastVar = type.isRight() ? input2.getName() : input1.getName();
		DataCharacteristics mcRdd = sec.getDataCharacteristics(rddVar);
		DataCharacteristics mcBc = sec.getDataCharacteristics(bcastVar);
		
		//get input rdd with preferred number of partitions to avoid unnecessary repartition
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable(rddVar,
			(requiresFlatMapFunction(type, mcBc) && requiresRepartitioning(
			type, mcRdd, mcBc, sec.getSparkContext().defaultParallelism())) ?
			getNumRepartitioning(type, mcRdd, mcBc) : -1, _outputEmpty);
		
		//investigate if a repartitioning - including a potential flip of broadcast and rdd 
		//inputs - is required to ensure moderately sized output partitions (2GB limitation)
		if( requiresFlatMapFunction(type, mcBc) &&
			requiresRepartitioning(type, mcRdd, mcBc, in1.getNumPartitions()) ) 
		{
			int numParts = getNumRepartitioning(type, mcRdd, mcBc);
			int numParts2 = getNumRepartitioning(type.getFlipped(), mcBc, mcRdd);
			if( numParts2 > numParts ) { //flip required
				type = type.getFlipped();
				rddVar = type.isRight() ? input1.getName() : input2.getName();
				bcastVar = type.isRight() ? input2.getName() : input1.getName();
				mcRdd = sec.getDataCharacteristics(rddVar);
				mcBc = sec.getDataCharacteristics(bcastVar);
				in1 = sec.getBinaryMatrixBlockRDDHandleForVariable(rddVar);
				LOG.warn("Mapmm: Switching rdd ('"+bcastVar+"') and broadcast ('"+rddVar+"') inputs "
						+ "for repartitioning because this allows better control of output partition "
						+ "sizes ("+numParts+" < "+numParts2+").");	
			}
		}
		
		//get inputs
		PartitionedBroadcast<MatrixBlock> in2 = sec.getBroadcastForVariable(bcastVar); 
		
		//empty input block filter
		if( !_outputEmpty )
			in1 = in1.filter(new FilterNonEmptyBlocksFunction());
		
		//execute mapmm and aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			JavaRDD<MatrixBlock> out = in1.map(new RDDMapMMFunction2(type, in2));
			MatrixBlock out2 = RDDAggregateUtils.sumStable(out);
			
			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);
		}
		else //MULTI_BLOCK or NONE
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			if( requiresFlatMapFunction(type, mcBc) ) {
				if( requiresRepartitioning(type, mcRdd, mcBc, in1.getNumPartitions()) ) {
					int numParts = getNumRepartitioning(type, mcRdd, mcBc);
					LOG.warn("Mapmm: Repartition input rdd '"+rddVar+"' from "+in1.getNumPartitions()+" to "
							+numParts+" partitions to satisfy size restrictions of output partitions.");
					in1 = in1.repartition(numParts);
				}
				out = in1.flatMapToPair( new RDDFlatMapMMFunction(type, in2) );
			}
			else if( preservesPartitioning(mcRdd, type) )
				out = in1.mapPartitionsToPair(new RDDMapMMPartitionFunction(type, in2), true);
			else
				out = in1.mapToPair( new RDDMapMMFunction(type, in2) );
			
			//empty output block filter
			if( !_outputEmpty )
				out = out.filter(new FilterNonEmptyBlocksFunction());
			
			if( _aggtype == SparkAggType.MULTI_BLOCK )
				out = RDDAggregateUtils.sumByKeyStable(out, false);
		
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			sec.addLineageBroadcast(output.getName(), bcastVar);
			
			//update output statistics if not inferred
			updateBinaryMMOutputDataCharacteristics(sec, true);
		}
	}

	private static boolean preservesPartitioning(DataCharacteristics mcIn, CacheType type )
	{
		if( type == CacheType.LEFT )
			return mcIn.dimsKnown() && mcIn.getRows() <= mcIn.getBlocksize();
		else // RIGHT
			return mcIn.dimsKnown() && mcIn.getCols() <= mcIn.getBlocksize();
	}
	
	/**
	 * Indicates if there is a need to apply a flatmap rdd operation because a single 
	 * input block creates multiple output blocks.
	 * 
	 * @param type cache type
	 * @param mcBc matrix characteristics
	 * @return true if single input block creates multiple output blocks
	 */
	private static boolean requiresFlatMapFunction( CacheType type, DataCharacteristics mcBc)
	{
		return    (type == CacheType.LEFT && mcBc.getRows() > mcBc.getBlocksize())
			   || (type == CacheType.RIGHT && mcBc.getCols() > mcBc.getBlocksize());
	}
	
	/**
	 * Indicates if there is a need to repartition the input RDD in order to increase the
	 * degree of parallelism or reduce the output partition size (e.g., Spark still has a
	 * 2GB limitation of partitions)
	 * 
	 * @param type cache type
	 * @param mcRdd rdd matrix characteristics
	 * @param mcBc ?
	 * @param numPartitions number of partitions
	 * @return true need to repartition input RDD
	 */
	private static boolean requiresRepartitioning(CacheType type, DataCharacteristics mcRdd, DataCharacteristics mcBc, int numPartitions ) {
		//note: as repartitioning requires data shuffling, we try to be very conservative here
		//approach: we repartition, if there is a "outer-product-like" mm (single block common dimension),
		//the size of output partitions (assuming dense) exceeds a size of 1GB 
		
		boolean isLeft = (type == CacheType.LEFT);
		boolean isOuter = isLeft ? 
				(mcRdd.getRows() <= mcRdd.getBlocksize()) :
				(mcRdd.getCols() <= mcRdd.getBlocksize());
		boolean isLargeOutput = (OptimizerUtils.estimatePartitionedSizeExactSparsity(isLeft?mcBc.getRows():mcRdd.getRows(),
				isLeft?mcRdd.getCols():mcBc.getCols(), mcRdd.getBlocksize(), 1.0) / numPartitions) > 1024*1024*1024; 
		return isOuter && isLargeOutput && mcRdd.dimsKnown() && mcBc.dimsKnown()
			&& numPartitions < getNumRepartitioning(type, mcRdd, mcBc);
	}

	/**
	 * Computes the number of target partitions for repartitioning input rdds in case of 
	 * outer-product-like mm.
	 * 
	 * @param type cache type
	 * @param mcRdd rdd matrix characteristics
	 * @param mcBc ?
	 * @return number of target partitions for repartitioning
	 */
	private static int getNumRepartitioning(CacheType type, DataCharacteristics mcRdd, DataCharacteristics mcBc ) {
		boolean isLeft = (type == CacheType.LEFT);
		long sizeOutput = (OptimizerUtils.estimatePartitionedSizeExactSparsity(isLeft?mcBc.getRows():mcRdd.getRows(),
				isLeft?mcRdd.getCols():mcBc.getCols(), mcRdd.getBlocksize(), 1.0)); 
		long numParts = sizeOutput / InfrastructureAnalyzer.getHDFSBlockSize();
		return (int)Math.min(numParts, (isLeft?mcRdd.getNumColBlocks():mcRdd.getNumRowBlocks()));
	}

	private static class RDDMapMMFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private final CacheType _type;
		private final AggregateBinaryOperator _op;
		private final PartitionedBroadcast<MatrixBlock> _pbc;
		
		public RDDMapMMFunction( CacheType type, PartitionedBroadcast<MatrixBlock> binput )
		{
			_type = type;
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			if( _type == CacheType.LEFT )
			{
				//get the right hand side matrix
				MatrixBlock left = _pbc.getBlock(1, (int)ixIn.getRowIndex());
				
				//execute matrix-vector mult
				OperationsOnMatrixValues.matMult(new MatrixIndexes(1,ixIn.getRowIndex()),
					left, ixIn, blkIn, ixOut, blkOut, _op);
			}
			else //if( _type == CacheType.RIGHT )
			{
				//get the right hand side matrix
				MatrixBlock right = _pbc.getBlock((int)ixIn.getColumnIndex(), 1);
				
				//execute matrix-vector mult
				OperationsOnMatrixValues.matMult(ixIn, blkIn,
					new MatrixIndexes(ixIn.getColumnIndex(),1), right, ixOut, blkOut, _op);
			}
			
			//output new tuple
			return new Tuple2<>(ixOut, blkOut);
		}
	}

	/**
	 * Similar to RDDMapMMFunction but with single output block 
	 */
	private static class RDDMapMMFunction2 implements Function<Tuple2<MatrixIndexes, MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = -2753453898072910182L;
		
		private final CacheType _type;
		private final AggregateBinaryOperator _op;
		private final PartitionedBroadcast<MatrixBlock> _pbc;
		
		public RDDMapMMFunction2( CacheType type, PartitionedBroadcast<MatrixBlock> binput )
		{
			_type = type;
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public MatrixBlock call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			if( _type == CacheType.LEFT )
			{
				//get the right hand side matrix
				MatrixBlock left = _pbc.getBlock(1, (int)ixIn.getRowIndex());
				
				//execute matrix-vector mult
				return OperationsOnMatrixValues.matMult( 
					left, blkIn, new MatrixBlock(), _op);
			}
			else //if( _type == CacheType.RIGHT )
			{
				//get the right hand side matrix
				MatrixBlock right = _pbc.getBlock((int)ixIn.getColumnIndex(), 1);
				
				//execute matrix-vector mult
				return OperationsOnMatrixValues.matMult(
					blkIn, right, new MatrixBlock(), _op);
			}
		}
	}

	private static class RDDMapMMPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1886318890063064287L;
	
		private final CacheType _type;
		private final AggregateBinaryOperator _op;
		private final PartitionedBroadcast<MatrixBlock> _pbc;
		
		public RDDMapMMPartitionFunction( CacheType type, PartitionedBroadcast<MatrixBlock> binput )
		{
			_type = type;
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
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
					MatrixBlock left = _pbc.getBlock(1, (int)ixIn.getRowIndex());
					
					//execute index preserving matrix multiplication
					OperationsOnMatrixValues.matMult(left, blkIn, blkOut, _op);
				}
				else //if( _type == CacheType.RIGHT )
				{
					//get the right hand side matrix
					MatrixBlock right = _pbc.getBlock((int)ixIn.getColumnIndex(), 1);

					//execute index preserving matrix multiplication
					OperationsOnMatrixValues.matMult(blkIn, right, blkOut, _op);
				}
			
				return new Tuple2<>(ixIn, blkOut);
			}
		}
	}

	private static class RDDFlatMapMMFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -6076256569118957281L;
		
		private final CacheType _type;
		private final AggregateBinaryOperator _op;
		private final PartitionedBroadcast<MatrixBlock> _pbc;
		
		public RDDFlatMapMMFunction( CacheType type, PartitionedBroadcast<MatrixBlock> binput )
		{
			_type = type;
			_pbc = binput;
			
			//created operator for reuse
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			if( _type == CacheType.LEFT ) {
				//for all matching left-hand-side blocks, returned as lazy iterator
				return IntStream.range(1, _pbc.getNumRowBlocks()+1).mapToObj(i ->
					new Tuple2<>(new MatrixIndexes(i, ixIn.getColumnIndex()),
					OperationsOnMatrixValues.matMult(_pbc.getBlock(i, (int)ixIn.getRowIndex()), blkIn,
						new MatrixBlock(), _op))).iterator();
			}
			else { //RIGHT
				//for all matching right-hand-side blocks, returned as lazy iterator
				return IntStream.range(1, _pbc.getNumColumnBlocks()+1).mapToObj(j ->
					new Tuple2<>(new MatrixIndexes(ixIn.getRowIndex(), j),
					OperationsOnMatrixValues.matMult(blkIn, _pbc.getBlock((int)ixIn.getColumnIndex(), j),
						new MatrixBlock(), _op))).iterator();
			}
		}
	}
}
