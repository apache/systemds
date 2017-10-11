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
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.lops.PickByCount.OperationTypes;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;

public class QuantilePickSPInstruction extends BinarySPInstruction {
	private OperationTypes _type = null;

	private QuantilePickSPInstruction(Operator op, CPOperand in, CPOperand out, OperationTypes type, boolean inmem,
			String opcode, String istr) {
		this(op, in, null, out, type, inmem, opcode, istr);
	}

	private QuantilePickSPInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
			boolean inmem, String opcode, String istr) {
		super(op, in, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.QPick;
		_type = type;
	}

	public static QuantilePickSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		//sanity check opcode
		if ( !opcode.equalsIgnoreCase("qpick") ) {
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
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input rdds
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		
		//NOTE: no difference between inmem/mr pick (see related cp instruction), but wrt w/ w/o weights
		//(in contrast to cp instructions, w/o weights does not materializes weights of 1)
		switch( _type ) {
			case VALUEPICK: {
				ScalarObject quantile = ec.getScalarInput(input2);
				double[] wt = getWeightedQuantileSummary(in, mc, quantile.getDoubleValue());
				ec.setScalarOutput(output.getName(), new DoubleObject(wt[3]));
				break;
			}
			
			case MEDIAN: {
				double[] wt = getWeightedQuantileSummary(in, mc, 0.5);
				ec.setScalarOutput(output.getName(), new DoubleObject(wt[3]));
				break;
			}
			
			case IQM: {
				double[] wt = getWeightedQuantileSummary(in, mc, 0.25, 0.75);
				long key25 = (long)Math.ceil(wt[1]);
				long key75 = (long)Math.ceil(wt[2]);
				JavaPairRDD<MatrixIndexes,MatrixBlock> out = in
					.filter(new FilterFunction(key25+1,key75,mc.getRowsPerBlock()))
					.mapToPair(new ExtractAndSumFunction(key25+1, key75, mc.getRowsPerBlock()));
				double sum = RDDAggregateUtils.sumStable(out).getValue(0, 0);
				double val = MatrixBlock.computeIQMCorrection(
					sum, wt[0], wt[3], wt[5], wt[4], wt[6]);
				ec.setScalarOutput(output.getName(), new DoubleObject(val));
				break;
			}
		
			default:
				throw new DMLRuntimeException("Unsupported qpick operation type: "+_type);
		}
	}
	
	/**
	 * Get a summary of weighted quantiles in in the following form:
	 * sum of weights, (keys of quantiles), (portions of quantiles), (values of quantiles)
	 * 
	 * @param w rdd containing values and optionally weights, sorted by value
	 * @param mc matrix characteristics
	 * @param quantiles one or more quantiles between 0 and 1.
	 * @return a summary of weighted quantiles
	 * @throws DMLRuntimeException 
	 */
	private static double[] getWeightedQuantileSummary(JavaPairRDD<MatrixIndexes,MatrixBlock> w, MatrixCharacteristics mc, Double... quantiles)
		throws DMLRuntimeException 
	{
		double[] ret = new double[3*quantiles.length + 1];
		if( mc.getCols()==2 ) //weighted 
		{
			//sort blocks (values sorted but blocks and partitions are not)
			w = w.sortByKey();
			
			//compute cumsum weights per partition
			//with assumption that partition aggregates fit into memory
			List<Tuple2<Integer,Double>> partWeights = w
				.mapPartitionsWithIndex(new SumWeightsFunction(), false).collect();
			
			//compute sum of weights
			ret[0] = partWeights.stream().mapToDouble(p -> p._2()).sum();
			
			//compute total cumsum and determine partitions
			double[] qdKeys = new double[quantiles.length];
			long[] qiKeys = new long[quantiles.length];
			int[] partitionIDs = new int[quantiles.length];
			double[] offsets = new double[quantiles.length];
			for( int i=0; i<quantiles.length; i++ ) {
				qdKeys[i] = quantiles[i]*ret[0];
				qiKeys[i] = (long)Math.ceil(qdKeys[i]);
			}
			double cumSum = 0;
			for( Tuple2<Integer,Double> psum : partWeights ) {
				double tmp = cumSum + psum._2();
				for(int i=0; i<quantiles.length; i++)
					if( tmp >= qiKeys[i] && partitionIDs[i] == 0 ) {
						partitionIDs[i] = psum._1();
						offsets[i] = cumSum;
					}
				cumSum = tmp;
			}
			
			//get keys and values for quantile cutoffs 
			List<Tuple2<Integer,double[]>> qVals = w
				.mapPartitionsWithIndex(new ExtractWeightedQuantileFunction(
					mc, qdKeys, qiKeys, partitionIDs, offsets), false).collect();
			for( Tuple2<Integer,double[]> qVal : qVals ) {
				ret[qVal._1()+1] = qVal._2()[0];
				ret[qVal._1()+quantiles.length+1] = qVal._2()[1];
				ret[qVal._1()+2*quantiles.length+1] = qVal._2()[2];
			}
		}
		else {
			ret[0] = mc.getRows();
			for( int i=0; i<quantiles.length; i++ ){
				ret[i+1] = quantiles[i] * mc.getRows();
				ret[i+quantiles.length+1] = Math.ceil(ret[i+1])-ret[i+1];
				ret[i+2*quantiles.length+1] = lookupKey(w, 
					(long)Math.ceil(ret[i+1]), mc.getRowsPerBlock());
			}
		}
		
		return ret;
	}

	private static double lookupKey(JavaPairRDD<MatrixIndexes,MatrixBlock> in, long key, int brlen)
		throws DMLRuntimeException
	{
		long rix = UtilFunctions.computeBlockIndex(key, brlen);
		long pos = UtilFunctions.computeCellInBlock(key, brlen);
		List<MatrixBlock> val = in.lookup(new MatrixIndexes(rix,1));
		if( val.isEmpty() )
			throw new DMLRuntimeException("Invalid key lookup in empty list.");
		MatrixBlock tmp = val.get(0);
		if( tmp.getNumRows() <= pos )
			throw new DMLRuntimeException("Invalid key lookup for " +
				pos + " in block of size " + tmp.getNumRows()+"x"+tmp.getNumColumns());
		return val.get(0).quickGetValue((int)pos, 0);
	}
	
	private static class FilterFunction implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> 
	{
		private static final long serialVersionUID = -8249102381116157388L;

		//boundary keys (inclusive)
		private long _minRowIndex;
		private long _maxRowIndex;
		
		public FilterFunction(long key25, long key75, int brlen) {
			_minRowIndex = UtilFunctions.computeBlockIndex(key25, brlen);
			_maxRowIndex = UtilFunctions.computeBlockIndex(key75, brlen);
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
		
		public ExtractAndSumFunction(long key25, long key75, int brlen)
		{
			_minRowIndex = UtilFunctions.computeBlockIndex(key25, brlen);
			_maxRowIndex = UtilFunctions.computeBlockIndex(key75, brlen);
			_minPos = UtilFunctions.computeCellInBlock(key25, brlen);
			_maxPos = UtilFunctions.computeCellInBlock(key75, brlen);
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
			ret.setValue(0, 0, (mb.getNumColumns()==1) ? 
				sum(mb, rl, ru) : sumWeighted(mb, rl, ru));
			return new Tuple2<>(new MatrixIndexes(1,1), ret);
		}
		
		private static double sum(MatrixBlock mb, int rl, int ru) {
			double sum = 0;
			for(int i=rl; i<ru; i++)
				sum += mb.quickGetValue(i, 0);
			return sum;
		}
		
		private static double sumWeighted(MatrixBlock mb, int rl, int ru) {
			double sum = 0;
			for(int i=rl; i<ru; i++)
				sum += mb.quickGetValue(i, 0)
					* mb.quickGetValue(i, 1);
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
		private final MatrixCharacteristics _mc;
		private final double[] _qdKeys;
		private final long[] _qiKeys;
		private final int[] _qPIDs;
		private final double[] _offsets;
		
		public ExtractWeightedQuantileFunction(MatrixCharacteristics mc, double[] qdKeys, long[] qiKeys, int[] qPIDs, double[] offsets) {
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
					double val = mb.quickGetValue(i, 1);
					for( int j=0; j<qlen; j++ ) {
						if( offset+val >= _qiKeys[qix[j]] ) {
							long pos = UtilFunctions.computeCellIndex(ix.getRowIndex(), _mc.getRowsPerBlock(), i);
							double posPart = offset+val - _qdKeys[qix[j]];
							ret.add(new Tuple2<>(qix[j], new double[]{pos, posPart, mb.quickGetValue(i, 0)}));
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
