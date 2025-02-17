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
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.data.BasicTensorBlock;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.functions.AggregateDropCorrectionFunction;
import org.apache.sysds.runtime.instructions.spark.functions.FilterDiagMatrixBlocksFunction;
import org.apache.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;

import scala.Tuple2;

public class AggregateUnarySPInstruction extends UnarySPInstruction {
	private SparkAggType _aggtype = null;
	private AggregateOperator _aop = null;

	protected AggregateUnarySPInstruction(SPType type, AggregateUnaryOperator auop, AggregateOperator aop, CPOperand in,
			CPOperand out, SparkAggType aggtype, String opcode, String istr) {
		super(type, auop, in, out, opcode, istr);
		_aggtype = aggtype;
		_aop = aop;
	}

	public static AggregateUnarySPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		SparkAggType aggtype = SparkAggType.valueOf(parts[3]);
		
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(opcode);
		CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(opcode);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrLoc.toString());
		return new AggregateUnarySPInstruction(SPType.AggregateUnary, aggun, aop, in1, out, aggtype, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		if (input1.getDataType() == Types.DataType.MATRIX) {
			processMatrixAggregate(ec);
		} else {
			processTensorAggregate(ec);
		}
	}

	private void processMatrixAggregate(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());

		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in;

		//filter input blocks for trace
		if( getOpcode().equalsIgnoreCase(Opcodes.UAKTRACE.toString()) )
			out = out.filter(new FilterDiagMatrixBlocksFunction());

		//execute unary aggregate operation
		AggregateUnaryOperator auop = (AggregateUnaryOperator)_optr;
		AggregateOperator aggop = _aop;

		//perform aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			if (ConfigurationManager.isMaxPrallelizeEnabled()) {
				//Trigger the chain of Spark operations and maintain a future to the result
				//TODO: Make memory for the future matrix block
				try {
					RDDAggregateTask task = new RDDAggregateTask(_optr, _aop, in, mc);
					Future<MatrixBlock> future_out = CommonThreadPool.getDynamicPool().submit(task);
					LineageItem li = !LineageCacheConfig.ReuseCacheType.isNone() ? getLineageItem(ec).getValue() : null;
					sec.setMatrixOutputAndLineage(output.getName(), future_out, li);
				}
				catch(Exception ex) {
					throw new DMLRuntimeException(ex);
				}
			}

			else {
				if( auop.sparseSafe )
					out = out.filter(new FilterNonEmptyBlocksFunction());

				JavaRDD<MatrixBlock> out2 = out.map(
						new RDDUAggFunction2(auop, mc.getBlocksize()));
				MatrixBlock out3 = RDDAggregateUtils.aggStable(out2, aggop);

				//drop correction after aggregation
				out3.dropLastRowsOrColumns(aggop.correction);

				//put output block into symbol table (no lineage because single block)
				//this also includes implicit maintenance of matrix characteristics
				sec.setMatrixOutput(output.getName(), out3);
			}
		}
		else //MULTI_BLOCK or NONE
		{
			if( _aggtype == SparkAggType.NONE ) {
				//in case of no block aggregation, we always drop the correction as well as
				//use a partitioning-preserving mapvalues
				out = out.mapValues(new RDDUAggValueFunction(auop, mc.getBlocksize()));
			}
			else if( _aggtype == SparkAggType.MULTI_BLOCK ) {
				//in case of multi-block aggregation, we always keep the correction
				out = out.mapToPair(new RDDUAggFunction(auop, mc.getBlocksize()));
				out = RDDAggregateUtils.aggByKeyStable(out, aggop, false);

				//drop correction after aggregation if required (aggbykey creates
				//partitioning, drop correction via partitioning-preserving mapvalues)
				if( auop.aggOp.existsCorrection() )
					out = out.mapValues( new AggregateDropCorrectionFunction(aggop) );
			}

			//put output RDD handle into symbol table
			updateUnaryAggOutputDataCharacteristics(sec, auop.indexFn);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
	}

	private void processTensorAggregate(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;

		//get input
		// TODO support DataTensor
		JavaPairRDD<TensorIndexes, TensorBlock> in = sec.getBinaryTensorBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<TensorIndexes, TensorBlock> out = in;

		// TODO: filter input blocks for trace
		//execute unary aggregate operation
		AggregateUnaryOperator auop = (AggregateUnaryOperator)_optr;
		AggregateOperator aggop = _aop;

		//perform aggregation if necessary and put output into symbol table
		if( _aggtype == SparkAggType.SINGLE_BLOCK )
		{
			// TODO filter non empty blocks if sparse safe
			JavaRDD<TensorBlock> out2 = out.map(new RDDUTensorAggFunction2(auop));
			TensorBlock out3 = RDDAggregateUtils.aggStableTensor(out2, aggop);

			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of data characteristics
			// TODO generalize to drop depending on location of correction
			// TODO support DataTensor
			TensorBlock out4 = new TensorBlock(out3.getValueType(), new int[]{1, 1});
			out4.set(0, 0, out3.get(0, 0));
			sec.setTensorOutput(output.getName(), out4);
		}
		else //MULTI_BLOCK or NONE
		{
			if( _aggtype == SparkAggType.NONE ) {
				//in case of no block aggregation, we always drop the correction as well as
				//use a partitioning-preserving mapvalues
				out = out.mapValues(new RDDUTensorAggValueFunction(auop));
			}
			else if( _aggtype == SparkAggType.MULTI_BLOCK ) {
				// TODO MULTI_BLOCK
				throw new DMLRuntimeException("Multi block spark aggregations are not supported for tensors yet.");
				/*
				//in case of multi-block aggregation, we always keep the correction
				out = out.mapToPair(new RDDUTensorAggFunction(auop, dc.getBlocksize(), dc.getBlocksize()));
				out = RDDAggregateUtils.aggByKeyStable(out, aggop, false);

				//drop correction after aggregation if required (aggbykey creates
				//partitioning, drop correction via partitioning-preserving mapvalues)
				if( auop.aggOp.correctionExists )
					out = out.mapValues( new AggregateDropCorrectionFunction(aggop) );
				 */
			}

			//put output RDD handle into symbol table
			updateUnaryAggOutputDataCharacteristics(sec, auop.indexFn);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
	}

	public SparkAggType getAggType() {
		return _aggtype;
	}

	private static class RDDUAggFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 2672082409287856038L;
		
		private AggregateUnaryOperator _op = null;
		private int _blen = -1;
		
		public RDDUAggFunction( AggregateUnaryOperator op, int blen ) {
			_op = op;
			_blen = blen;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();
			
			//unary aggregate operation (always keep the correction)
			OperationsOnMatrixValues.performAggregateUnary(
				ixIn, blkIn, ixOut, blkOut, _op, _blen);
			
			//output new tuple
			return new Tuple2<>(ixOut, blkOut);
		}
	}

	/**
	 * Similar to RDDUAggFunction but single output block.
	 */
	public static class RDDUAggFunction2 implements Function<Tuple2<MatrixIndexes, MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = 2672082409287856038L;
		
		private AggregateUnaryOperator _op = null;
		private int _blen = -1;
		
		public RDDUAggFunction2( AggregateUnaryOperator op, int blen ) {
			_op = op;
			_blen = blen;
			_blen = blen;
		}
		
		@Override
		public MatrixBlock call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			//unary aggregate operation (always keep the correction)
			return (MatrixBlock) arg0._2.aggregateUnaryOperations(
					_op, new MatrixBlock(), _blen, arg0._1());
		}
	}

	/**
	 * Similar to RDDUAggFunction but single output block.
	 */
	public static class RDDUTensorAggFunction2 implements Function<Tuple2<TensorIndexes, TensorBlock>, TensorBlock>
	{
		private static final long serialVersionUID = -6258769067791011763L;

		private AggregateUnaryOperator _op = null;

		public RDDUTensorAggFunction2( AggregateUnaryOperator op ) {
			_op = op;
		}

		@Override
		public TensorBlock call(Tuple2<TensorIndexes, TensorBlock> arg0 )
				throws Exception
		{
			//unary aggregate operation (always keep the correction)
			// TODO support DataTensor
			return new TensorBlock(arg0._2.getBasicTensor().aggregateUnaryOperations(_op, new BasicTensorBlock()));
		}
	}

	private static class RDDUAggValueFunction implements Function<MatrixBlock, MatrixBlock>
	{
		private static final long serialVersionUID = 5352374590399929673L;
		
		private AggregateUnaryOperator _op = null;
		private int _blen = -1;
		private MatrixIndexes _ix = null;
		
		public RDDUAggValueFunction( AggregateUnaryOperator op, int blen ) {
			_op = op;
			_blen = blen;
			
			_ix = new MatrixIndexes(1,1);
		}
		
		@Override
		public MatrixBlock call( MatrixBlock arg0 ) 
			throws Exception 
		{
			MatrixBlock blkOut = new MatrixBlock();
			
			//unary aggregate operation
			arg0.aggregateUnaryOperations(_op, blkOut, _blen, _ix, true);
			
			//output new tuple
			return blkOut;
		}
	}

	private static class RDDUTensorAggValueFunction implements Function<TensorBlock, TensorBlock>
	{
		private static final long serialVersionUID = -968274963539513423L;

		private AggregateUnaryOperator _op = null;

		public RDDUTensorAggValueFunction(AggregateUnaryOperator op)
		{
			_op = op;
		}

		@Override
		public TensorBlock call(TensorBlock arg0 )
				throws Exception
		{
			// TODO support DataTensor
			BasicTensorBlock blkOut = new BasicTensorBlock();

			//unary aggregate operation
			arg0.getBasicTensor().aggregateUnaryOperations(_op, blkOut);

			//always drop correction since no aggregation
			// TODO generalize to drop depending on location of correction
			TensorBlock out = new TensorBlock(blkOut.getValueType(), new int[]{1, 1});
			out.set(0, 0, blkOut.get(0, 0));

			//output new tuple
			return out;
		}
	}

	private static class RDDAggregateTask implements Callable<MatrixBlock>
	{
		Operator _optr;
		AggregateOperator _aop;
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in;
		DataCharacteristics _mc;

		RDDAggregateTask(Operator optr, AggregateOperator aop, JavaPairRDD<MatrixIndexes,
			MatrixBlock> input, DataCharacteristics dc) {
			_optr = optr;
			_aop = aop;
			_in = input;
			_mc = dc;
		}

		@Override
		public MatrixBlock call() {
			AggregateUnaryOperator auop = (AggregateUnaryOperator)_optr;
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = _in;
			if( auop.sparseSafe )
				out = out.filter(new FilterNonEmptyBlocksFunction());

			JavaRDD<MatrixBlock> out2 = out.map(
				new RDDUAggFunction2(auop, _mc.getBlocksize()));
			MatrixBlock out3 = RDDAggregateUtils.aggStable(out2, _aop);

			//drop correction after aggregation
			out3.dropLastRowsOrColumns(_aop.correction);
			return out3;
		}
	}
}
