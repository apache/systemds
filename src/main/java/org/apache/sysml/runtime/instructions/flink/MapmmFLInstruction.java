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

package org.apache.sysml.runtime.instructions.flink;


import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import org.apache.sysml.hops.AggBinaryOp.SparkAggType;
import org.apache.sysml.lops.MapMult;
import org.apache.sysml.lops.MapMult.CacheType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.FlinkExecutionContext;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.flink.functions.FilterNonEmptyBlocksFunction;
import org.apache.sysml.runtime.instructions.flink.functions.MatrixMultiplicationFunctions;
import org.apache.sysml.runtime.instructions.flink.functions.RichFlatMapBroadcastFunction;
import org.apache.sysml.runtime.instructions.flink.functions.RichMapBroadcastFunction;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;

import java.util.HashMap;
import java.util.Map;

/**
 * TODO: we need to reason about multiple broadcast variables for chains of mapmults (sum of operations until cleanup)
 */
public class MapmmFLInstruction extends BinaryFLInstruction {

	private CacheType _type = null;
	private boolean _outputEmpty = true;
	private SparkAggType _aggtype;

	public MapmmFLInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, CacheType type,
							  boolean outputEmpty, SparkAggType aggtype, String opcode, String istr) {
		super(op, in1, in2, out, opcode, istr);
		_fltype = FLINSTRUCTION_TYPE.MAPMM;

		_type = type;
		_outputEmpty = outputEmpty;
		_aggtype = aggtype;
	}

	/**
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MapmmFLInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if (opcode.equalsIgnoreCase(MapMult.OPCODE)) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			CacheType type = CacheType.valueOf(parts[4]);
			boolean outputEmpty = Boolean.parseBoolean(parts[5]);
			SparkAggType aggtype = SparkAggType.valueOf(parts[6]);

			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new MapmmFLInstruction(aggbin, in1, in2, out, type, outputEmpty, aggtype, opcode, str);
		} else {
			throw new DMLRuntimeException("MapmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}

	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		FlinkExecutionContext flec = (FlinkExecutionContext) ec;

		String datasetVar = (_type == CacheType.LEFT) ? input2.getName() : input1.getName();
		String bcastVar = (_type == CacheType.LEFT) ? input1.getName() : input2.getName();
		MatrixCharacteristics mcDataSet = flec.getMatrixCharacteristics(datasetVar);
		MatrixCharacteristics mcBc = flec.getMatrixCharacteristics(bcastVar);

		//get inputs
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> in1 = flec.getBinaryBlockDataSetHandleForVariable(datasetVar);

		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> in2 = flec.getBinaryBlockDataSetHandleForVariable(bcastVar);

		//empty input block filter
		if (!_outputEmpty)
			in1 = in1.filter(new FilterNonEmptyBlocksFunction());

		//execute mapmult instruction
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = null;
		if (requiresFlatMapFunction(_type, mcBc)) {
			out = in1.flatMap(new DataSetFlatMapMMFunction(_type)).withBroadcastSet(in2, "bcastVar");
		} else if (preservesPartitioning(mcDataSet, _type)) {
			out = in1.flatMap(new DataSetMapMMPartitionFunction(_type)).withBroadcastSet(in2, "bcastVar");
		} else {
			out = in1.map(new DataSetMapMMFunction(_type)).withBroadcastSet(in2, "bcastVar");
		}

		//empty output block filter
		if (!_outputEmpty)
			out = out.filter(new FilterNonEmptyBlocksFunction());


		//perform aggregation if necessary and put output into symbol table
		if (_aggtype == SparkAggType.SINGLE_BLOCK) {
			MatrixBlock out2 = DataSetAggregateUtils.sumStable1(out);

			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			flec.setMatrixOutput(output.getName(), out2);
		} else //MULTI_BLOCK or NONE
		{
			if (_aggtype == SparkAggType.MULTI_BLOCK) {
				out = out.groupBy(new MatrixMultiplicationFunctions.MatrixIndexesSelector())
						.reduce(new MatrixMultiplicationFunctions.SumMatrixBlocksStable());
			}

			//put output DataSet handle into symbol table
			flec.setDataSetHandleForVariable(output.getName(), out);
			flec.addLineageDataSet(output.getName(), datasetVar);

			//update output statistics if not inferred
			updateBinaryMMOutputMatrixCharacteristics(flec, true);
		}
	}

	/**
	 * @param mcIn
	 * @param type
	 * @return
	 */
	private static boolean preservesPartitioning(MatrixCharacteristics mcIn, CacheType type) {
		if (type == CacheType.LEFT)
			return mcIn.dimsKnown() && mcIn.getRows() <= mcIn.getRowsPerBlock();
		else // RIGHT
			return mcIn.dimsKnown() && mcIn.getCols() <= mcIn.getColsPerBlock();
	}

	/**
	 * @param type
	 * @param mcBc
	 * @return
	 */
	private static boolean requiresFlatMapFunction(CacheType type, MatrixCharacteristics mcBc) {
		return (type == CacheType.LEFT && mcBc.getRows() > mcBc.getRowsPerBlock())
				|| (type == CacheType.RIGHT && mcBc.getCols() > mcBc.getColsPerBlock());
	}

	/**
	 *
	 *
	 */
	private static class DataSetMapMMFunction
			extends RichMapBroadcastFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {
		private CacheType _type = null;

		//created operator for reuse
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator _op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		public DataSetMapMMFunction(CacheType type) {
			_type = type;
		}

		public DataSetMapMMFunction() {

		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> map(Tuple2<MatrixIndexes, MatrixBlock> arg0)
				throws Exception {
			MatrixIndexes ixIn = arg0.f0;
			MatrixBlock blkIn = arg0.f1;

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();

			if (_type == CacheType.LEFT) {

				//get the right hand side matrix
				MatrixBlock left = _pbc.get(1L).get(ixIn.getRowIndex());

				//execute matrix-vector mult

				OperationsOnMatrixValues.performAggregateBinary(
						new MatrixIndexes(1L, ixIn.getRowIndex()), left, ixIn, blkIn, ixOut, blkOut, _op);
			} else //if( _type == CacheType.RIGHT )
			{
				//get the right hand side matrix
				MatrixBlock right = _pbc.get(ixIn.getColumnIndex()).get(1L);

				//execute matrix-vector mult
				OperationsOnMatrixValues.performAggregateBinary(
						ixIn, blkIn, new MatrixIndexes(ixIn.getColumnIndex(), 1L), right, ixOut, blkOut, _op);
			}

			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}

	/**
	 *
	 */
	private static class DataSetMapMMPartitionFunction
			extends RichFlatMapBroadcastFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {
		private CacheType _type = null;

		//created operator for reuse
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator _op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		public DataSetMapMMPartitionFunction(CacheType type) {
			_type = type;
		}

		public DataSetMapMMPartitionFunction() {

		}

		@Override
		public void flatMap(Tuple2<MatrixIndexes, MatrixBlock> arg0, Collector<Tuple2<MatrixIndexes, MatrixBlock>> out)
				throws Exception {
			MatrixIndexes ixIn = arg0.f0;
			MatrixBlock blkIn = arg0.f1;
			MatrixBlock blkOut = new MatrixBlock();

			if (_type == CacheType.LEFT) {
				//get the right hand side matrix
				MatrixBlock left = _pbc.get(1L).get(ixIn.getRowIndex());
				//execute index preserving matrix multiplication
				left.aggregateBinaryOperations(left, blkIn, blkOut, _op);


			} else //if( _type == CacheType.RIGHT )
			{
				MatrixBlock right = _pbc.get(ixIn.getColumnIndex()).get(1L);

				//execute index preserving matrix multiplication
				blkIn.aggregateBinaryOperations(blkIn, right, blkOut, _op);

			}

			out.collect(new Tuple2<MatrixIndexes, MatrixBlock>(ixIn, blkOut));
		}
	}

	/**
	 *
	 *
	 */
	private static class DataSetFlatMapMMFunction
			extends RichFlatMapBroadcastFunction<Tuple2<MatrixIndexes, MatrixBlock>,
			Tuple2<MatrixIndexes, MatrixBlock>> {
		private CacheType _type = null;

		//created operator for reuse
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator _op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		public DataSetFlatMapMMFunction(CacheType type) {
			_type = type;
		}

		public DataSetFlatMapMMFunction() {

		}

		@Override
		public void flatMap(Tuple2<MatrixIndexes, MatrixBlock> arg0, Collector<Tuple2<MatrixIndexes, MatrixBlock>> out)
				throws Exception {
			MatrixIndexes ixIn = arg0.f0;
			MatrixBlock blkIn = arg0.f1;

			if (_type == CacheType.LEFT) {
				//for all matching left-hand-side blocks
				for (Map.Entry<Long, HashMap<Long, MatrixBlock>> i : _pbc.entrySet()) {
					MatrixBlock left = i.getValue().get(ixIn.getRowIndex());

					MatrixIndexes ixOut = new MatrixIndexes();
					MatrixBlock blkOut = new MatrixBlock();

					//execute matrix-vector mult
					OperationsOnMatrixValues.performAggregateBinary(
							new MatrixIndexes(i.getKey(), ixIn.getRowIndex()), left, ixIn, blkIn, ixOut, blkOut, _op);

					out.collect(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut));
				}
			} else //if( _type == CacheType.RIGHT )
			{
				//for all matching right-hand-side blocks
				HashMap<Long, MatrixBlock> blocklist = _pbc.get(ixIn.getColumnIndex());

				for (Map.Entry<Long, MatrixBlock> j : blocklist.entrySet()) {
					//get the right hand side matrix
					MatrixBlock right = j.getValue();
					MatrixIndexes ixOut = new MatrixIndexes();
					MatrixBlock blkOut = new MatrixBlock();

					//execute matrix-vector mult
					OperationsOnMatrixValues.performAggregateBinary(
							ixIn, blkIn, new MatrixIndexes(ixIn.getColumnIndex(), j.getKey()), right, ixOut, blkOut,
							_op);

					out.collect(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut));
				}
			}
		}
	}
}
