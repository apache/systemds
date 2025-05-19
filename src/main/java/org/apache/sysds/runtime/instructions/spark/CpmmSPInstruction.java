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

import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.apache.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction2;
import org.apache.sysds.runtime.instructions.spark.functions.ReorgMapFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;

import scala.Tuple2;

/**
 * Cpmm: cross-product matrix multiplication operation (distributed matrix multiply
 * by join over common dimension and subsequent aggregation of partial results).
 * 
 * NOTE: There is additional optimization potential by preventing aggregation for a single
 * block on the common dimension. However, in such a case we would never pick cpmm because
 * this would result in a degree of parallelism of 1.
 * 
 */
public class CpmmSPInstruction extends AggregateBinarySPInstruction {
	private final boolean _outputEmptyBlocks;
	private final SparkAggType _aggtype;
	
	private CpmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
		boolean outputEmptyBlocks, SparkAggType aggtype, String opcode, String istr) {
		super(SPType.CPMM, op, in1, in2, out, opcode, istr);
		_outputEmptyBlocks = outputEmptyBlocks;
		_aggtype = aggtype;
	}

	public static CpmmSPInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if ( !opcode.equalsIgnoreCase(Opcodes.CPMM.toString()))
			throw new DMLRuntimeException("CpmmSPInstruction.parseInstruction(): Unknown opcode " + opcode);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		AggregateBinaryOperator aggbin = InstructionUtils.getMatMultOperator(1);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[4]);
		SparkAggType aggtype = SparkAggType.valueOf(parts[5]);
		return new CpmmSPInstruction(aggbin, in1, in2, out, outputEmptyBlocks, aggtype, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get rdd inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable(input2.getName());
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		
		if( !_outputEmptyBlocks || _aggtype == SparkAggType.SINGLE_BLOCK
			|| mc1.isNoEmptyBlocks() || mc2.isNoEmptyBlocks() ) {
			//prune empty blocks of ultra-sparse matrices
			in1 = in1.filter(new FilterNonEmptyBlocksFunction());
			in2 = in2.filter(new FilterNonEmptyBlocksFunction());
		}
		
		if( SparkUtils.isHashPartitioned(in1) //ZIPMM-like CPMM
			&& mc1.getNumRowBlocks()==1 && mc2.getCols()==1 )
		//note: if the major input is hash-partitioned and it's a matrix-vector
		//multiply, avoid the index mapping to preserve the partitioning similar
		//to a ZIPMM but with different transpose characteristics
		{
			if (ConfigurationManager.isMaxPrallelizeEnabled()) {
				try {
					CpmmMatrixVectorTask task = new CpmmMatrixVectorTask(in1, in2);
					Future<MatrixBlock> future_out = CommonThreadPool.getDynamicPool().submit(task);
					LineageItem li = !LineageCacheConfig.ReuseCacheType.isNone() ? getLineageItem(ec).getValue() : null;
					sec.setMatrixOutputAndLineage(output.getName(), future_out, li);
				}
				catch(Exception ex) {
					throw new DMLRuntimeException(ex);
				}
			}
			else {
				JavaRDD<MatrixBlock> out = in1.join(in2.mapToPair(new ReorgMapFunction(Opcodes.TRANSPOSE.toString()))).values().map(new Cpmm2MultiplyFunction()).filter(new FilterNonEmptyBlocksFunction2());
				MatrixBlock out2 = RDDAggregateUtils.sumStable(out);

				//put output block into symbol table (no lineage because single block)
				//this also includes implicit maintenance of matrix characteristics
				sec.setMatrixOutput(output.getName(), out2);
			}
		}
		else //GENERAL CPMM
		{
			//compute preferred join degree of parallelism
			int numPreferred = getPreferredParJoin(mc1, mc2, in1.getNumPartitions(), in2.getNumPartitions());
			int numPartJoin = Math.min(getMaxParJoin(mc1, mc2), numPreferred);
			
			//process core cpmm matrix multiply 
			JavaPairRDD<Long, IndexedMatrixValue> tmp1 = in1.mapToPair(new CpmmIndexFunction(true));
			JavaPairRDD<Long, IndexedMatrixValue> tmp2 = in2.mapToPair(new CpmmIndexFunction(false));

			//process cpmm aggregation and handle outputs
			if( _aggtype == SparkAggType.SINGLE_BLOCK )
			{
				if (ConfigurationManager.isMaxPrallelizeEnabled()) {
					try {
						CpmmMatrixMatrixTask task = new CpmmMatrixMatrixTask(in1, in2, numPartJoin);
						Future<MatrixBlock> future_out = CommonThreadPool.getDynamicPool().submit(task);
						sec.setMatrixOutput(output.getName(), future_out);
					}
					catch(Exception ex) { throw new DMLRuntimeException(ex); }
				}
				else {
					JavaPairRDD<MatrixIndexes, MatrixBlock> out = tmp1
						.join(tmp2, numPartJoin)                // join over common dimension
						.mapToPair(new CpmmMultiplyFunction()); // compute block multiplications
					//prune empty blocks and aggregate all results
					out = out.filter(new FilterNonEmptyBlocksFunction());
					MatrixBlock out2 = RDDAggregateUtils.sumStable(out);

					//put output block into symbol table (no lineage because single block)
					//this also includes implicit maintenance of matrix characteristics
					sec.setMatrixOutput(output.getName(), out2);
				}

			}
			else
			{ //DEFAULT: MULTI_BLOCK
				JavaPairRDD<MatrixIndexes,MatrixBlock> out = tmp1
					.join(tmp2, numPartJoin)                // join over common dimension
					.mapToPair(new CpmmMultiplyFunction()); // compute block multiplications
				if( !_outputEmptyBlocks || mc1.isNoEmptyBlocks() || mc2.isNoEmptyBlocks() )
					out = out.filter(new FilterNonEmptyBlocksFunction());
				out = RDDAggregateUtils.sumByKeyStable(out, false);
				
				//put output RDD handle into symbol table
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), input1.getName());
				sec.addLineageRDD(output.getName(), input2.getName());
				
				//update output statistics if not inferred
				updateBinaryMMOutputDataCharacteristics(sec, true);
			}
		}
	}

	public SparkAggType getAggType() {
		return _aggtype;
	}

	private static int getPreferredParJoin(DataCharacteristics mc1, DataCharacteristics mc2, int numPar1, int numPar2) {
		int defPar = SparkExecutionContext.getDefaultParallelism(true);
		int maxParIn = Math.max(numPar1, numPar2);
		int maxSizeIn = SparkUtils.getNumPreferredPartitions(mc1) +
			SparkUtils.getNumPreferredPartitions(mc2);
		int tmp = (mc1.dimsKnown(true) && mc2.dimsKnown(true)) ? 
			Math.max(maxSizeIn, maxParIn) : maxParIn;
		return (tmp > defPar/2) ? Math.max(tmp, defPar) : tmp;
	}
	
	private static int getMaxParJoin(DataCharacteristics mc1, DataCharacteristics mc2) {
		return mc1.colsKnown() ? (int)mc1.getNumColBlocks() :
			mc2.rowsKnown() ? (int)mc2.getNumRowBlocks() :
			Integer.MAX_VALUE;
	}

	private static class CpmmIndexFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, Long, IndexedMatrixValue>
	{
		private static final long serialVersionUID = -1187183128301671162L;
		private final boolean _left;
		
		public CpmmIndexFunction( boolean left ) {
			_left = left;
		}
		
		@Override
		public Tuple2<Long, IndexedMatrixValue> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			IndexedMatrixValue value = new IndexedMatrixValue(arg0._1(), arg0._2());
			Long key = _left ? arg0._1.getColumnIndex() : arg0._1.getRowIndex();
			return new Tuple2<>(key, value);
		}
	}

	private static class CpmmMultiplyFunction implements PairFunction<Tuple2<Long, Tuple2<IndexedMatrixValue,IndexedMatrixValue>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -2009255629093036642L;
		private AggregateBinaryOperator _op = null;

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<Long, Tuple2<IndexedMatrixValue, IndexedMatrixValue>> arg0)
			throws Exception
		{
			if( _op == null ) { //lazy operator construction
				_op = InstructionUtils.getMatMultOperator(1);
			}
			
			MatrixBlock blkIn1 = (MatrixBlock)arg0._2()._1().getValue();
			MatrixBlock blkIn2 = (MatrixBlock)arg0._2()._2().getValue();
			MatrixIndexes ixOut = new MatrixIndexes();
			
			//core block matrix multiplication 
			MatrixBlock blkOut = OperationsOnMatrixValues
				.matMult(blkIn1, blkIn2, new MatrixBlock(), _op);
			
			//return target block
			ixOut.setIndexes(arg0._2()._1().getIndexes().getRowIndex(),
				arg0._2()._2().getIndexes().getColumnIndex());
			return new Tuple2<>( ixOut, blkOut );
		}
	}
	
	private static class Cpmm2MultiplyFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock>
	{
		private static final long serialVersionUID = -3718880362385713416L;
		private AggregateBinaryOperator _op = null;
		private ReorgOperator _rop = null;
		
		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0) throws Exception {
			 //lazy operator construction
			if( _op == null ) {
				_op = InstructionUtils.getMatMultOperator(1);
				_rop = new ReorgOperator(SwapIndex.getSwapIndexFnObject());
			}
			//prepare inputs, including transpose of right-hand-side
			MatrixBlock in1 = arg0._1();
			MatrixBlock in2 = arg0._2().reorgOperations(_rop, new MatrixBlock(), 0, 0, 0);
			//core block matrix multiplication
			return OperationsOnMatrixValues.matMult(in1, in2, new MatrixBlock(), _op);
		}
	}

	private static class CpmmMatrixVectorTask implements Callable<MatrixBlock>
	{
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in1;
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in2;

		CpmmMatrixVectorTask(JavaPairRDD<MatrixIndexes, MatrixBlock> in1, JavaPairRDD<MatrixIndexes, MatrixBlock> in2) {
			_in1 = in1;
			_in2 = in2;
		}
		@Override
		public MatrixBlock call() {
				JavaRDD<MatrixBlock> out = _in1
				.join(_in2.mapToPair(new ReorgMapFunction(Opcodes.TRANSPOSE.toString())))
				.values().map(new Cpmm2MultiplyFunction())
				.filter(new FilterNonEmptyBlocksFunction2());
			return RDDAggregateUtils.sumStable(out);
		}
	}

	private static class CpmmMatrixMatrixTask implements Callable<MatrixBlock>
	{
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in1;
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in2;
		int _numPartJoin;

		CpmmMatrixMatrixTask(JavaPairRDD<MatrixIndexes, MatrixBlock> in1, JavaPairRDD<MatrixIndexes, MatrixBlock> in2, int nPartJoin) {
			_in1 = in1;
			_in2 = in2;
			_numPartJoin = nPartJoin;
		}
		@Override
		public MatrixBlock call() {
			//process core cpmm matrix multiply
			JavaPairRDD<Long, IndexedMatrixValue> tmp1 = _in1.mapToPair(new CpmmIndexFunction(true));
			JavaPairRDD<Long, IndexedMatrixValue> tmp2 = _in2.mapToPair(new CpmmIndexFunction(false));
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = tmp1
				.join(tmp2, _numPartJoin)                // join over common dimension
				.mapToPair(new CpmmMultiplyFunction()); // compute block multiplications

			//prune empty blocks and aggregate all results
			out = out.filter(new FilterNonEmptyBlocksFunction());
			return RDDAggregateUtils.sumStable(out);
		}
	}
}
