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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.PickByCount.OperationTypes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class QuantilePickSPInstruction extends BinarySPInstruction {
	private OperationTypes _type = null;

	private QuantilePickSPInstruction(Operator op, CPOperand in, CPOperand out, OperationTypes type, boolean inmem,
			String opcode, String istr) {
		this(op, in, null, out, type, inmem, opcode, istr);
	}

	private QuantilePickSPInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
			boolean inmem, String opcode, String istr) {
		super(SPType.QPick, op, in, in2, out, opcode, istr);
		_type = type;
	}

	public static QuantilePickSPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		//sanity check opcode
		if ( !opcode.equalsIgnoreCase(Opcodes.QPICK.toString()) ) {
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantilePickCPInstruction: " + str);
		}
		
		//instruction parsing
		if( parts.length == 4 ) {
			//instructions of length 4 originate from unary - mr-iqm
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.IQM;
			return new QuantilePickSPInstruction(null, in1, in2, out, ptype, false, opcode, str);
		}
		else if( parts.length == 5 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[2]);
			OperationTypes ptype = OperationTypes.valueOf(parts[3]);
			boolean inmem = Boolean.parseBoolean(parts[4]);
			return new QuantilePickSPInstruction(null, in1, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 6 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.valueOf(parts[4]);
			boolean inmem = Boolean.parseBoolean(parts[5]);
			return new QuantilePickSPInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		
		return null;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input rdds
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());
		
		//NOTE: no difference between inmem/mr pick (see related cp instruction), but wrt w/ w/o weights
		//(in contrast to cp instructions, w/o weights does not materializes weights of 1)
		switch(_type) {
			case VALUEPICK: {
				if(input2.isScalar()) {
					double picked = pickQuantileValues(in, mc,
						new double[] {ec.getScalarInput(input2).getDoubleValue()})[0];
					ec.setScalarOutput(output.getName(), new DoubleObject(picked));
				}
				else {
					double[] values = pickQuantileValues(in, mc,
						DataConverter.convertToDoubleVector(ec.getMatrixInput(input2.getName())));
					ec.releaseMatrixInput(input2.getName());
					MatrixBlock out = new MatrixBlock(values.length, 1, false);
					for(int i = 0; i < values.length; i++)
						out.set(i, 0, values[i]);
					ec.setMatrixOutput(output.getName(), out);
				}
				break;
			}

			case MEDIAN: {
				double median = pickQuantileValues(in, mc, new double[] {0.5})[0];
				ec.setScalarOutput(output.getName(), new DoubleObject(median));
				break;
			}

			case IQM: {
				double val = computeIqm(in, mc);
				ec.setScalarOutput(output.getName(), new DoubleObject(val));
				break;
			}

			default:
				throw new DMLRuntimeException("Unsupported qpick operation type: " + _type);
		}
	}

	/**
	 * Pick one R quantile type 7 value per requested probability. Used by VALUEPICK / MEDIAN. Two-column input is a
	 * weighted sequence, treated as an expanded sorted sequence of length sum(weights) and picked with the same h/lo/
	 * hi/g formula against cumulative weights.
	 */
	private static double[] pickQuantileValues(JavaPairRDD<MatrixIndexes, MatrixBlock> w, DataCharacteristics mc,
		double[] quantiles) {
		final int blen = mc.getBlocksize();
		final double[] values = new double[quantiles.length];
		if(mc.getCols() == 2) {
			final JavaPairRDD<MatrixIndexes, MatrixBlock> sorted = w.sortByKey();
			final List<Tuple2<Integer, Double>> partWeights = sorted
				.mapPartitionsWithIndex(new SumWeightsFunction(), false).collect();
			final long sumWt = Math.round(partWeights.stream().mapToDouble(p -> p._2()).sum());
			// Two keys per quantile (lo, hi) for type-7 interpolation; qdKey == qiKey since posPart is unused here.
			final int nk = 2 * quantiles.length;
			final double[] qdKeys = new double[nk];
			final long[] qiKeys = new long[nk];
			final double[] gs = new double[quantiles.length];
			for(int i = 0; i < quantiles.length; i++) {
				final double[] r = MatrixBlock.computeQuantileRank(sumWt, quantiles[i]);
				qiKeys[2 * i] = (long) r[0];
				qiKeys[2 * i + 1] = (long) r[1];
				qdKeys[2 * i] = r[0];
				qdKeys[2 * i + 1] = r[1];
				gs[i] = r[2];
			}
			final double[][] triples = extractWeightedTriples(sorted, mc, partWeights, qdKeys, qiKeys);
			for(int i = 0; i < quantiles.length; i++) {
				final double loVal = triples[2 * i][2];
				// hi == lo covers the p == 1 clamp.
				values[i] = (gs[i] == 0.0 || qiKeys[2 * i + 1] == qiKeys[2 * i]) ? loVal : (1.0 - gs[i]) * loVal +
					gs[i] * triples[2 * i + 1][2];
			}
		}
		else {
			final long N = mc.getRows();
			for(int i = 0; i < quantiles.length; i++) {
				final double[] r = MatrixBlock.computeQuantileRank(N, quantiles[i]);
				final long lo = (long) r[0], hi = (long) r[1];
				final double g = r[2];
				final double loVal = lookupKey(w, lo, blen);
				values[i] = (1.0 - g) * loVal + g * lookupKey(w, hi, blen);
			}
		}
		return values;
	}

	/**
	 * Compute the interquartile mean: trimmed mean of values between the ceil-based q25 and q75 ranks, with fractional
	 * boundary corrections applied via {@link MatrixBlock#computeIQMCorrection}. IQM uses raw ceil-based boundaries
	 * (not R type 7) so the middle-range sum and boundary portions align with the closed-form correction formula.
	 */
	private static double computeIqm(JavaPairRDD<MatrixIndexes, MatrixBlock> in, DataCharacteristics mc) {
		final int blen = mc.getBlocksize();
		final double sumWt, q25Position, q75Position, q25Portion, q75Portion, q25Value, q75Value;
		if(mc.getCols() == 2) {
			final JavaPairRDD<MatrixIndexes, MatrixBlock> sorted = in.sortByKey();
			final List<Tuple2<Integer, Double>> partWeights = sorted
				.mapPartitionsWithIndex(new SumWeightsFunction(), false).collect();
			sumWt = partWeights.stream().mapToDouble(p -> p._2()).sum();
			final double[] qdKeys = {0.25 * sumWt, 0.75 * sumWt};
			final long[] qiKeys = {(long) Math.ceil(qdKeys[0]), (long) Math.ceil(qdKeys[1])};
			final double[][] triples = extractWeightedTriples(sorted, mc, partWeights, qdKeys, qiKeys);
			// For weighted data q25/q75Position is the sorted matrix row index (from extract), not
			// 0.25 * sumWt in the expanded sequence — the IQM filter runs on RDD row-block coordinates.
			q25Position = triples[0][0];
			q75Position = triples[1][0];
			q25Portion = triples[0][1];
			q75Portion = triples[1][1];
			q25Value = triples[0][2];
			q75Value = triples[1][2];
		}
		else {
			final long N = mc.getRows();
			sumWt = N;
			q25Position = 0.25 * N;
			q75Position = 0.75 * N;
			q25Portion = Math.ceil(q25Position) - q25Position;
			q75Portion = Math.ceil(q75Position) - q75Position;
			q25Value = lookupKey(in, (long) Math.ceil(q25Position), blen);
			q75Value = lookupKey(in, (long) Math.ceil(q75Position), blen);
		}
		final long key25 = (long) Math.ceil(q25Position);
		final long key75 = (long) Math.ceil(q75Position);
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = in.filter(new FilterFunction(key25 + 1, key75, blen))
			.mapToPair(new ExtractAndSumFunction(key25 + 1, key75, blen));
		double sum = RDDAggregateUtils.sumStable(out).get(0, 0);
		return MatrixBlock.computeIQMCorrection(sum, sumWt, q25Portion, q25Value, q75Portion, q75Value);
	}

	/**
	 * Locate the partition holding each ceil-based key by scanning cumulative per-partition weights, then invoke
	 * {@link ExtractWeightedQuantileFunction} to fetch the (position, posPart, value) triples. Returns a nk-length
	 * array indexed by the caller's key index, one triple per key.
	 */
	private static double[][] extractWeightedTriples(JavaPairRDD<MatrixIndexes, MatrixBlock> sorted,
		DataCharacteristics mc, List<Tuple2<Integer, Double>> partWeights, double[] qdKeys, long[] qiKeys) {
		final int nk = qiKeys.length;
		final int[] partitionIDs = new int[nk];
		final double[] offsets = new double[nk];
		double cumSum = 0;
		for(Tuple2<Integer, Double> psum : partWeights) {
			final double tmp = cumSum + psum._2();
			for(int i = 0; i < nk; i++)
				if(tmp >= qiKeys[i] && partitionIDs[i] == 0) {
					partitionIDs[i] = psum._1();
					offsets[i] = cumSum;
				}
			cumSum = tmp;
		}
		final List<Tuple2<Integer, double[]>> qVals = sorted.mapPartitionsWithIndex(
			new ExtractWeightedQuantileFunction(mc, qdKeys, qiKeys, partitionIDs, offsets), false).collect();
		final double[][] triples = new double[nk][];
		for(Tuple2<Integer, double[]> qVal : qVals)
			triples[qVal._1()] = qVal._2();
		return triples;
	}

	private static double lookupKey(JavaPairRDD<MatrixIndexes,MatrixBlock> in, long key, int blen) {
		long rix = UtilFunctions.computeBlockIndex(key, blen);
		long pos = UtilFunctions.computeCellInBlock(key, blen);
		List<MatrixBlock> val = in.lookup(new MatrixIndexes(rix,1));
		if( val.isEmpty() )
			throw new DMLRuntimeException("Invalid key lookup in empty list.");
		MatrixBlock tmp = val.get(0);
		if( tmp.getNumRows() <= pos )
			throw new DMLRuntimeException("Invalid key lookup for " +
				pos + " in block of size " + tmp.getNumRows()+"x"+tmp.getNumColumns());
		return val.get(0).get((int)pos, 0);
	}

	public OperationTypes getOperationType() {
		return _type;
	}

	private static class FilterFunction implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> 
	{
		private static final long serialVersionUID = -8249102381116157388L;

		//boundary keys (inclusive)
		private long _minRowIndex;
		private long _maxRowIndex;
		
		public FilterFunction(long key25, long key75, int blen) {
			_minRowIndex = UtilFunctions.computeBlockIndex(key25, blen);
			_maxRowIndex = UtilFunctions.computeBlockIndex(key75, blen);
		}

		@Override
		public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception 
		{
			long rowIndex = arg0._1().getRowIndex();
			return (rowIndex>=_minRowIndex && rowIndex<=_maxRowIndex);
		}
	}

	private static class ExtractAndSumFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -584044441055250489L;
		
		//boundary keys (inclusive)
		private long _minRowIndex;
		private long _maxRowIndex;
		private int _minPos;
		private int _maxPos;
		
		public ExtractAndSumFunction(long key25, long key75, int blen)
		{
			_minRowIndex = UtilFunctions.computeBlockIndex(key25, blen);
			_maxRowIndex = UtilFunctions.computeBlockIndex(key75, blen);
			_minPos = UtilFunctions.computeCellInBlock(key25, blen);
			_maxPos = UtilFunctions.computeCellInBlock(key75, blen);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			int rl = (ix.getRowIndex() == _minRowIndex) ? _minPos : 0;
			int ru = (ix.getRowIndex() == _maxRowIndex) ? _maxPos+1 : mb.getNumRows();
			MatrixBlock ret = new MatrixBlock(1,2,false);
			ret.set(0, 0, (mb.getNumColumns()==1) ? 
				sum(mb, rl, ru) : sumWeighted(mb, rl, ru));
			return new Tuple2<>(new MatrixIndexes(1,1), ret);
		}
		
		private static double sum(MatrixBlock mb, int rl, int ru) {
			double sum = 0;
			for(int i=rl; i<ru; i++)
				sum += mb.get(i, 0);
			return sum;
		}
		
		private static double sumWeighted(MatrixBlock mb, int rl, int ru) {
			double sum = 0;
			for(int i=rl; i<ru; i++)
				sum += mb.get(i, 0)
					* mb.get(i, 1);
			return sum;
 		}
	}

	private static class SumWeightsFunction implements Function2<Integer,Iterator<Tuple2<MatrixIndexes,MatrixBlock>>,Iterator<Tuple2<Integer, Double>>> 
	{
		private static final long serialVersionUID = 7169831202450745373L;

		@Override
		public Iterator<Tuple2<Integer, Double>> call(Integer v1, Iterator<Tuple2<MatrixIndexes, MatrixBlock>> v2)
			throws Exception 
		{
			//aggregate partition weights (in sorted order)
			double sum = 0;
			while( v2.hasNext() )
				sum += v2.next()._2().sumWeightForQuantile();
			
			//return tuple for partition aggregate
			return Arrays.asList(new Tuple2<>(v1,sum)).iterator();
		}
	}
	
	private static class ExtractWeightedQuantileFunction implements Function2<Integer,Iterator<Tuple2<MatrixIndexes,MatrixBlock>>,Iterator<Tuple2<Integer, double[]>>> 
	{
		private static final long serialVersionUID = 4879975971050093739L;
		private final DataCharacteristics _mc;
		private final double[] _qdKeys;
		private final long[] _qiKeys;
		private final int[] _qPIDs;
		private final double[] _offsets;
		
		public ExtractWeightedQuantileFunction(DataCharacteristics mc, double[] qdKeys, long[] qiKeys, int[] qPIDs, double[] offsets) {
			_mc = mc;
			_qdKeys = qdKeys;
			_qiKeys = qiKeys;
			_qPIDs = qPIDs;
			_offsets = offsets;
		}

		@Override
		public Iterator<Tuple2<Integer, double[]>> call(Integer v1, Iterator<Tuple2<MatrixIndexes, MatrixBlock>> v2) 
			throws Exception 
		{
			//early abort for unnecessary partitions
			if( !ArrayUtils.contains(_qPIDs, v1) )
				return Collections.emptyIterator();
			
			//determine which quantiles are active
			int qlen = (int)Arrays.stream(_qPIDs).filter(i -> i==v1).count();
			int[] qix = new int[qlen];
			for(int i=0, pos=0; i<_qPIDs.length; i++)
				if( _qPIDs[i]==v1 )
					qix[pos++] = i;
			double offset = _offsets[qix[0]];
			
			//iterate over blocks and determine quantile positions
			ArrayList<Tuple2<Integer,double[]>> ret = new ArrayList<>();
			while( v2.hasNext() ) {
				Tuple2<MatrixIndexes, MatrixBlock> tmp = v2.next();
				MatrixIndexes ix = tmp._1();
				MatrixBlock mb = tmp._2();
				for( int i=0; i<mb.getNumRows(); i++ ) {
					double val = mb.get(i, 1);
					for( int j=0; j<qlen; j++ ) {
						if( offset+val >= _qiKeys[qix[j]] ) {
							long pos = UtilFunctions.computeCellIndex(ix.getRowIndex(), _mc.getBlocksize(), i);
							double posPart = offset+val - _qdKeys[qix[j]];
							ret.add(new Tuple2<>(qix[j], new double[]{pos, posPart, mb.get(i, 0)}));
							_qiKeys[qix[j]] = Long.MAX_VALUE;
						}
					}
					offset += val;
				}
			}
			return ret.iterator();
		}
	}
}
