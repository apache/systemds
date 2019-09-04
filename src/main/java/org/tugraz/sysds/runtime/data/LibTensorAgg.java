/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tugraz.sysds.runtime.data;

import org.apache.commons.lang.NotImplementedException;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.functionobjects.ValueFunction;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.util.CommonThreadPool;
import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

public class LibTensorAgg {
	private enum AggType {
		SUM,
		INVALID,
	}

	/**
	 * Check if a aggregation fulfills the constraints to be split to multiple threads.
	 *
	 * @param in the tensor block to be aggregated
	 * @param k  the number of threads
	 * @return true if aggregation should be done on multiple threads, false otherwise
	 */
	public static boolean satisfiesMultiThreadingConstraints(BasicTensorBlock in, int k) {
		// TODO more conditions depending on operation
		return k > 1 && in._vt != Types.ValueType.BOOLEAN;
	}

	/**
	 * Aggregate a tensor-block with the given unary operator.
	 *
	 * @param in   the input tensor block
	 * @param out  the output tensor block containing the aggregated result
	 * @param uaop the unary operation to apply
	 */
	public static void aggregateUnaryTensor(BasicTensorBlock in, BasicTensorBlock out, AggregateUnaryOperator uaop) {
		AggType aggType = getAggType(uaop);
		// TODO filter empty input blocks (incl special handling for sparse-unsafe operations)
		if (in.isEmpty(false)) {
			aggregateUnaryTensorEmpty(in, out, aggType);
			return;
		}
		int numThreads = uaop.getNumThreads();
		if (satisfiesMultiThreadingConstraints(in, numThreads)) {
			try {
				ExecutorService pool = CommonThreadPool.get(numThreads);
				ArrayList<AggTask> tasks = new ArrayList<>();
				ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(in.getDim(0), numThreads, false);
				for (int i = 0, lb = 0; i < blklens.size(); lb += blklens.get(i), i++) {
					tasks.add(new PartialAggTask(in, out, aggType, uaop, lb, lb + blklens.get(i)));
				}
				pool.invokeAll(tasks);
				pool.shutdown();
				//aggregate partial results
				out.copy(((PartialAggTask) tasks.get(0)).getResult()); //for init
				for (int i = 1; i < tasks.size(); i++)
					aggregateFinalResult(uaop.aggOp, out, ((PartialAggTask) tasks.get(i)).getResult());
			} catch (Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		} else {
			// Actually a complete aggregation
			if (!in.isSparse()) {
				aggregateUnaryTensorPartial(in, out, aggType, uaop.aggOp.increOp.fn, 0, in.getDim(0));
			} else {
				throw new NotImplementedException("Tensor aggregation not supported for sparse tensors.");
			}
		}
		// TODO change to sparse if worth it
	}

	/**
	 * Aggregate a empty tensor-block with a unary operator.
	 *
	 * @param in     the tensor-block to aggregate
	 * @param out    the resulting tensor-block
	 * @param optype the operation to apply
	 */
	private static void aggregateUnaryTensorEmpty(BasicTensorBlock in, BasicTensorBlock out, AggType optype) {
		// TODO implement for other optypes
		double val;
		if (optype == AggType.SUM) {
			val = 0;
		} else {
			val = Double.NaN;
		}
		out.set(new int[]{0, 0}, val);
	}

	/**
	 * Core incremental tensor aggregate (ak+) as used for uack+ and acrk+.
	 * Embedded correction values.
	 *
	 * @param in partial aggregation
	 * @param aggVal partial aggregation, also output (in will be added to this)
	 * @param aop aggregation operator
	 */
	public static void aggregateBinaryTensor(BasicTensorBlock in, BasicTensorBlock aggVal, AggregateOperator aop) {
		//check validity
		if (in.getLength() != aggVal.getLength()) {
			throw new DMLRuntimeException("Binary tensor aggregation requires consistent numbers of cells (" +
					Arrays.toString(in._dims) + ", " + Arrays.toString(aggVal._dims) + ").");
		}

		//core aggregation
		// TODO support indexfn that are not reduce all
		// TODO support for all shapes
		if (!aop.correctionExists) {
			if (aop.increOp.fn instanceof Plus) {
				int[] first = new int[in.getNumDims()];
				switch (in.getValueType()) {
					case INT64:
						aggVal.set(first, (Long)in.get(first) + (Long)aggVal.get(first));
						break;
					case INT32:
						aggVal.set(first, (Integer)in.get(first) + (Integer)aggVal.get(first));
						break;
					default:
						aggVal.set(0, 0, in.get(0, 0) + aggVal.get(0, 0));
						break;
				}
			}
			else {
				throw new DMLRuntimeException("Binary aggregation of this type not supported for tensors yet");
			}
		}
		else {
			throw new DMLRuntimeException("Corrections not supported for tensors yet");
		}
	}

	/**
	 * Get the aggregation type from the unary operator.
	 *
	 * @param op the unary operator
	 * @return the aggregation type
	 */
	private static AggType getAggType(AggregateUnaryOperator op) {
		ValueFunction vfn = op.aggOp.increOp.fn;
		// sum
		if (vfn instanceof Plus)
			return AggType.SUM;
		return AggType.INVALID;
	}

	/**
	 * Determines whether the unary operator is supported.
	 *
	 * @param op the unary operator to check
	 * @return true if the operator is supported, false otherwise
	 */
	public static boolean isSupportedUnaryAggregateOperator(AggregateUnaryOperator op) {
		AggType type = getAggType(op);
		return type != AggType.INVALID;
	}

	/**
	 * Aggregate a subset of rows of a dense tensor block.
	 *
	 * @param in      the tensor block to aggregate
	 * @param out     the aggregation result with correction
	 * @param aggtype the type of aggregation to use
	 * @param fn      the function to use
	 * @param rl      the lower index of rows to use
	 * @param ru      the upper index of rows to use (exclusive)
	 */
	private static void aggregateUnaryTensorPartial(BasicTensorBlock in, BasicTensorBlock out, AggType aggtype, ValueFunction fn,
	                                                int rl, int ru) {
		//note: due to corrections, even the output might be a large dense block
		if (aggtype == AggType.SUM) {
			// TODO handle different index functions
			sum(in, out, (Plus) fn, rl, ru);
		}
		// TODO other aggregations
	}

	/**
	 * Add two partial aggregations together.
	 *
	 * @param aop     the aggregation operator
	 * @param out     the tensor-block which contains partial result and should be increased to contain sum of both results
	 * @param partout the tensor-block which contains partial result and should be added to other partial result
	 */
	private static void aggregateFinalResult(AggregateOperator aop, BasicTensorBlock out, BasicTensorBlock partout) {
		//TODO special handling for mean where the final aggregate operator
		// is not equals to the partial aggregate operator
		//incremental aggregation of final results
		if (!aop.correctionExists)
			out.incrementalAggregate(aop, partout);
		else
			throw new NotImplementedException();
		//out.binaryOperationsInPlace(laop.increOp, partout);
	}

	private static void sum(BasicTensorBlock in, BasicTensorBlock out, Plus plus, int rl, int ru) {
		// TODO: SparseBlock
		if (in.isSparse()) {
			throw new DMLRuntimeException("Sparse aggregation not implemented for Tensor");
		}
		switch (in.getValueType()) {
			case BOOLEAN: {
				//TODO switch to no-op nnz meta data once available
				out.set(0, 0, in.getDenseBlock().countNonZeros());
				break;
			}
			case STRING: {
				throw new DMLRuntimeException("Sum over string tensor is not supported.");
			}
			case FP64:
			case FP32: {
				DenseBlock a = in.getDenseBlock();
				double sum = 0;
				for (int r = rl; r < ru; r++) {
					for (int c = 0; c < a.getCumODims(0); c++) {
						sum = plus.execute(sum, a.get(r, c));
					}
				}
				out.set(0, 0, sum);
				break;
			}
			case INT64:
			case INT32: {
				DenseBlock a = in.getDenseBlock();
				long sum = 0;
				int[] ix = new int[a.numDims()];
				for (int r = rl; r < ru; r++) {
					ix[0] = r;
					for (int c = 0; c < a.getCumODims(0); c++) {
						ix[ix.length - 1] = c; // linear scan whole row
						sum += a.getLong(ix);
					}
				}
				out.set(new int[out.getNumDims()], sum);
				break;
			}
			case UNKNOWN:
				throw new NotImplementedException();
		}
	}

	// TODO maybe merge this, and other parts, with `LibMatrixAgg`
	private static abstract class AggTask implements Callable<Object> {}

	private static class PartialAggTask extends AggTask {
		private BasicTensorBlock _in;
		private BasicTensorBlock _ret;
		private AggType _aggtype;
		private AggregateUnaryOperator _uaop;
		private int _rl;
		private int _ru;

		protected PartialAggTask(BasicTensorBlock in, BasicTensorBlock ret, AggType aggtype, AggregateUnaryOperator uaop, int rl, int ru) {
			_in = in;
			_ret = ret;
			_aggtype = aggtype;
			_uaop = uaop;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Object call() {
			//thead-local allocation for partial aggregation
			_ret = new BasicTensorBlock(_ret._vt, new int[]{_ret.getDim(0), _ret.getDim(1)});
			_ret.allocateDenseBlock();

			aggregateUnaryTensorPartial(_in, _ret, _aggtype, _uaop.aggOp.increOp.fn, _rl, _ru);
			//TODO recompute non-zeros of partial result
			return null;
		}

		public BasicTensorBlock getResult() {
			return _ret;
		}
	}
}
