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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperatorTypes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.utils.Hash;
import scala.Tuple2;

public class AggregateUnarySketchSPInstruction extends UnarySPInstruction {
	private AggBinaryOp.SparkAggType aggtype;
	private CountDistinctOperator op;

	protected AggregateUnarySketchSPInstruction(Operator op, CPOperand in, CPOperand out,
		AggBinaryOp.SparkAggType aggtype, String opcode, String instr) {
		super(SPType.AggregateUnarySketch, op, in, out, opcode, instr);
		this.op = (CountDistinctOperator) super.getOperator();
		this.aggtype = aggtype;
	}

	public static AggregateUnarySketchSPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];

		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		AggBinaryOp.SparkAggType aggtype = AggBinaryOp.SparkAggType.valueOf(parts[3]);

		CountDistinctOperator cdop = null;
		if(opcode.equals(Opcodes.UACD.toString())) {
			cdop = new CountDistinctOperator(CountDistinctOperatorTypes.COUNT, Types.Direction.RowCol,
				ReduceAll.getReduceAllFnObject(), Hash.HashType.LinearHash);
		}
		else if(opcode.equals(Opcodes.UACDR.toString())) {
			throw new NotImplementedException("uacdr has not been implemented yet");
		}
		else if(opcode.equals(Opcodes.UACDC.toString())) {
			throw new NotImplementedException("uacdc has not been implemented yet");
		}
		else if(opcode.equals(Opcodes.UACDAP.toString())) {
			cdop = new CountDistinctOperator(CountDistinctOperatorTypes.KMV, Types.Direction.RowCol,
				ReduceAll.getReduceAllFnObject(), Hash.HashType.LinearHash);
		}
		else if(opcode.equals(Opcodes.UACDAPR.toString())) {
			cdop = new CountDistinctOperator(CountDistinctOperatorTypes.KMV, Types.Direction.Row,
				ReduceCol.getReduceColFnObject(), Hash.HashType.LinearHash);
		}
		else if(opcode.equals(Opcodes.UACDAPC.toString())) {
			cdop = new CountDistinctOperator(CountDistinctOperatorTypes.KMV, Types.Direction.Col,
				ReduceRow.getReduceRowFnObject(), Hash.HashType.LinearHash);
		}
		else {
			throw new DMLException("Unrecognized opcode: " + opcode);
		}

		return new AggregateUnarySketchSPInstruction(cdop, in1, out, aggtype, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(input1.getDataType() == Types.DataType.MATRIX) {
			processMatrixSketch(ec);
		}
		else {
			processTensorSketch(ec);
		}
	}

	private void processMatrixSketch(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		// get input
		JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = in;

		// dir = RowCol and (dim1() > 1000 || dim2() > 1000)
		if(aggtype == AggBinaryOp.SparkAggType.SINGLE_BLOCK) {

			// Create a single sketch and derive approximate count distinct from the sketch
			JavaRDD<CorrMatrixBlock> out1 = out.map(new AggregateUnarySketchCreateFunction(this.op));

			// Using fold() instead of reduce() for stable aggregation
			// Instantiating CorrMatrixBlock mutable buffer with empty matrix block so that it can be serialized properly
			CorrMatrixBlock out2 = out1.fold(new CorrMatrixBlock(new MatrixBlock()),
				new AggregateUnarySketchUnionAllFunction(this.op));

			MatrixBlock out3 = LibMatrixCountDistinct.countDistinctValuesFromSketch(this.op, out2);

			// put output block into symbol table (no lineage because single block)
			// this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out3);
		}
		else {

			if(aggtype != AggBinaryOp.SparkAggType.NONE && aggtype != AggBinaryOp.SparkAggType.MULTI_BLOCK) {
				throw new DMLRuntimeException(String.format("Unsupported aggregation type: %s", aggtype));
			}

			JavaPairRDD<MatrixIndexes, MatrixBlock> out1;
			JavaPairRDD<MatrixIndexes, CorrMatrixBlock> out2;
			JavaPairRDD<MatrixIndexes, MatrixBlock> out3;

			// dir = Row || Col || RowCol and (dim1() <= 1000 || dim2() <= 1000)
			if(aggtype == AggBinaryOp.SparkAggType.NONE) {
				// Input matrix is small enough for a single index, so there is no need to execute index function.
				// Reuse the CreateCombinerFunction(), although there is no need to merge values within the same
				// partition, or combiners across partitions for that matter.
				out2 = out.mapValues(new AggregateUnarySketchCreateCombinerFunction(this.op));

				// aggType = MULTI_BLOCK: dir = Row || Col and (dim1() > 1000 || dim2() > 1000)
			}
			else {
				// Execute index function to group rows/columns together based on aggregation direction
				out1 = out.mapToPair(new RowColGroupingFunction(this.op));

				// Create sketch per index
				out2 = out1.combineByKey(new AggregateUnarySketchCreateCombinerFunction(this.op),
					new AggregateUnarySketchMergeValueFunction(this.op),
					new AggregateUnarySketchMergeCombinerFunction(this.op));
			}

			out3 = out2.mapValues(new CalculateAggregateSketchFunction(this.op));

			updateUnaryAggOutputDataCharacteristics(sec, this.op.indexFn);

			// put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out3);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
	}

	private void processTensorSketch(ExecutionContext ec) {
		throw new NotImplementedException("Aggregate sketch instruction for tensors has not been implemented yet.");
	}

	public AggBinaryOp.SparkAggType getAggType() {
		return aggtype;
	}

	private static class AggregateUnarySketchCreateFunction
		implements Function<Tuple2<MatrixIndexes, MatrixBlock>, CorrMatrixBlock> {
		private static final long serialVersionUID = 7295176181965491548L;
		private CountDistinctOperator op;

		public AggregateUnarySketchCreateFunction(CountDistinctOperator op) {
			this.op = op;
		}

		@Override
		public CorrMatrixBlock call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			this.op.indexFn.execute(ixIn, ixOut);

			return LibMatrixCountDistinct.createSketch(this.op, blkIn);
		}
	}

	private static class AggregateUnarySketchUnionAllFunction
		implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> {
		private static final long serialVersionUID = -3799519241499062936L;
		private CountDistinctOperator op;

		public AggregateUnarySketchUnionAllFunction(CountDistinctOperator op) {
			this.op = op;
		}

		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, CorrMatrixBlock arg1) throws Exception {

			// Input matrix blocks must have corresponding sketch metadata
			if(arg0.getCorrection() == null && arg1.getCorrection() == null) {
				throw new DMLRuntimeException("Corrupt sketch: metadata is missing");
			}

			if((arg0.getValue().getNumRows() == 0 && arg0.getValue().getNumColumns() == 0) ||
				arg0.getCorrection() == null) {
				arg0.set(arg1.getValue(), arg1.getCorrection());
				return arg0;
			}
			else if((arg1.getValue().getNumRows() == 0 && arg1.getValue().getNumColumns() == 0) ||
				arg1.getCorrection() == null) {
				return arg0;
			}

			return LibMatrixCountDistinct.unionSketch(this.op, arg0, arg1);
		}
	}

	private static class RowColGroupingFunction
		implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = -3456633769452405482L;
		private CountDistinctOperator _op;

		public RowColGroupingFunction(CountDistinctOperator op) {
			this._op = op;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes idxIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes idxOut = new MatrixIndexes();
			MatrixBlock blkOut = blkIn; // Do not create sketch yet
			this._op.indexFn.execute(idxIn, idxOut);

			return new Tuple2<>(idxOut, blkOut);
		}
	}

	private static class AggregateUnarySketchCreateCombinerFunction implements Function<MatrixBlock, CorrMatrixBlock> {
		private static final long serialVersionUID = 8997980606986435297L;
		private final CountDistinctOperator op;

		private AggregateUnarySketchCreateCombinerFunction(CountDistinctOperator op) {
			this.op = op;
		}

		@Override
		public CorrMatrixBlock call(MatrixBlock arg0) throws Exception {

			return LibMatrixCountDistinct.createSketch(this.op, arg0);
		}
	}

	private static class AggregateUnarySketchMergeValueFunction
		implements Function2<CorrMatrixBlock, MatrixBlock, CorrMatrixBlock> {
		private static final long serialVersionUID = -7006864809860460549L;
		private CountDistinctOperator op;

		public AggregateUnarySketchMergeValueFunction(CountDistinctOperator op) {
			this.op = op;
		}

		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, MatrixBlock arg1) throws Exception {
			CorrMatrixBlock arg1WithCorr = LibMatrixCountDistinct.createSketch(this.op, arg1);
			return LibMatrixCountDistinct.unionSketch(this.op, arg0, arg1WithCorr);
		}
	}

	private static class AggregateUnarySketchMergeCombinerFunction
		implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> {
		private static final long serialVersionUID = 172215143740379070L;
		private CountDistinctOperator op;

		public AggregateUnarySketchMergeCombinerFunction(CountDistinctOperator op) {
			this.op = op;
		}

		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, CorrMatrixBlock arg1) throws Exception {
			return LibMatrixCountDistinct.unionSketch(this.op, arg0, arg1);
		}
	}

	private static class CalculateAggregateSketchFunction implements Function<CorrMatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = 7504873483231717138L;
		private CountDistinctOperator op;

		public CalculateAggregateSketchFunction(CountDistinctOperator op) {
			this.op = op;
		}

		@Override
		public MatrixBlock call(CorrMatrixBlock arg0) throws Exception {
			return LibMatrixCountDistinct.countDistinctValuesFromSketch(this.op, arg0);
		}
	}
}
