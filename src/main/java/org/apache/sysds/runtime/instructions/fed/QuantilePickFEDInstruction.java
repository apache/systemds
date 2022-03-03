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

package org.apache.sysds.runtime.instructions.fed;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.lops.PickByCount.OperationTypes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

@SuppressWarnings("unchecked")
public class QuantilePickFEDInstruction extends BinaryFEDInstruction {

	private final OperationTypes _type;

	private QuantilePickFEDInstruction(Operator op, CPOperand in, CPOperand out, OperationTypes type, boolean inmem,
			String opcode, String istr) {
		this(op, in, null, out, type, inmem, opcode, istr);
	}

	private QuantilePickFEDInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
			boolean inmem, String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.QPick, op, in, in2, out, opcode, istr, fedOut);
		_type = type;
	}

	private QuantilePickFEDInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
		boolean inmem, String opcode, String istr) {
		this(op, in, in2, out, type, inmem, opcode, istr, FederatedOutput.NONE);
	}

	public OperationTypes getQPickType() {
		return _type;
	}

	public static QuantilePickFEDInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if ( !opcode.equalsIgnoreCase("qpick") )
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantilePickCPInstruction: " + str);
		//instruction parsing
		if( parts.length == 4 ) {
			//instructions of length 4 originate from unary - mr-iqm
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.IQM;
			boolean inmem = false;
			return new QuantilePickFEDInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 5 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[2]);
			OperationTypes ptype = OperationTypes.valueOf(parts[3]);
			boolean inmem = Boolean.parseBoolean(parts[4]);
			return new QuantilePickFEDInstruction(null, in1, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 6 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.valueOf(parts[4]);
			boolean inmem = Boolean.parseBoolean(parts[5]);
			return new QuantilePickFEDInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		return null;
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(ec.getMatrixObject(input1).isFederated(FederationMap.FType.COL) || ec.getMatrixObject(input1).isFederated(FederationMap.FType.FULL))
			processColumnQPick(ec);
		else
			processRowQPick(ec);
	}

	public <T> void processRowQPick(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMap = in.getFedMapping();
		boolean average = _type == OperationTypes.MEDIAN;

		double[] quantiles = input2 != null ? (input2.isMatrix() ? ec.getMatrixInput(input2).getDenseBlockValues() :
			input2.isScalar() ? new double[] {ec.getScalarInput(input2).getDoubleValue()} : null) :
			(average ? new double[] {0.5} : _type == OperationTypes.IQM ? new double[] {0.25, 0.75} : null);

		if (input2 != null && input2.isMatrix())
			ec.releaseMatrixInput(input2.getName());

		// Find min and max
		long varID = FederationUtils.getNextFedDataID();
		List<double[]> minMax = new ArrayList<>();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new QuantilePickFEDInstruction.MinMax(data.getVarID()))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				double[] rangeMinMax = (double[]) response.getData()[0];
				minMax.add(rangeMinMax);

				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		// Find weights sum, min and max
		double globalMin = Double.MAX_VALUE, globalMax = Double.MIN_VALUE, vectorLength = in.getNumColumns() == 2 ? 0 : in.getNumRows(), sumWeights = 0.0;
		for(double[] values : minMax) {
			globalMin = Math.min(globalMin, values[0]);
			globalMax = Math.max(globalMax, values[1]);
			if(in.getNumColumns() == 2)
				vectorLength += values[2];
			sumWeights += values[3];
		}

		// Average for median
		average = average && (in.getNumColumns() == 2 ? sumWeights : in.getNumRows()) % 2 == 0;

		// If multiple quantiles take first histogram and reuse bins, otherwise recursively get bin with result
		int numBuckets = 256; // (int) Math.round(in.getNumRows() / 2.0);
		int quantileIndex = quantiles != null && quantiles.length == 1 ? (int) Math.round(vectorLength * quantiles[0]) : -1;

		T ret = createHistogram(in, (int) vectorLength, globalMin, globalMax, numBuckets, quantileIndex, average);

		// Compute and set results
		if(quantiles != null && quantiles.length > 1) {
			computeMultipleQuantiles(ec, in, (Map<ImmutablePair<Double, Double>, Integer>) ret, quantiles, (int) vectorLength, varID, _type);
		} else
			getSingleQuantileResult(ret, ec, fedMap, varID, average, false, (int) vectorLength);
	}

	private <T> void computeMultipleQuantiles(ExecutionContext ec, MatrixObject in, Map<ImmutablePair<Double, Double>, Integer> buckets, double[] quantiles, int vectorLength, long varID, OperationTypes type) {
		MatrixBlock out = new MatrixBlock(quantiles.length, 1, false);
		ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>>[] bucketsWithIndex = new ImmutableTriple[quantiles.length];

		// Find bins with each quantile for first histogram
		int sizeBeforeTmp = 0, sizeBefore = 0, countFoundBins = 0;
		for(Map.Entry<ImmutablePair<Double, Double>, Integer> entry : buckets.entrySet()) {
			sizeBeforeTmp += entry.getValue();

			for(int i = 0; i < quantiles.length; i++) {
				int quantileIndex = (int) Math.round(vectorLength * quantiles[i]);
				ImmutablePair<Double, Double> bucketWithQ = null;

				if(quantileIndex > sizeBefore && quantileIndex <= sizeBeforeTmp) {
					bucketWithQ = entry.getKey();
					bucketsWithIndex[i] = new ImmutableTriple<>(quantileIndex == 1 ? 1 : quantileIndex - sizeBefore, entry.getValue(), bucketWithQ);
					countFoundBins++;
				}
			}

			sizeBefore = sizeBeforeTmp;
			if(countFoundBins == quantiles.length)
				break;
		}

		// Find each quantile bin recursively
		Map<Integer, T> retBuckets = new HashMap<>();

		double left = 0, right = 0;
		for(int i = 0; i < bucketsWithIndex.length; i++) {
			int nextNumBuckets = bucketsWithIndex[i].middle < 100 ? bucketsWithIndex[i].middle * 2 : (int) Math.round(bucketsWithIndex[i].middle / 2.0);
			T hist = createHistogram(in, vectorLength, bucketsWithIndex[i].right.left, bucketsWithIndex[i].right.right, nextNumBuckets, bucketsWithIndex[i].left, false);

			if(_type == OperationTypes.IQM) {
				left = i == 0 ? hist instanceof ImmutablePair ?  ((ImmutablePair<Double, Double>)hist).right : (Double) hist : left;
				right = i == 1 ? hist instanceof ImmutablePair ? ((ImmutablePair<Double, Double>)hist).left : (Double) hist : right;
			} else {
				if(hist instanceof ImmutablePair)
					retBuckets.put(i, hist); // set value if returned double instead of bin
				else
					out.setValue(i, 0, (Double) hist);
			}
		}

		if(type == OperationTypes.IQM) {
			ImmutablePair<Double, Double> IQMRange = new ImmutablePair<>(left, right);
			getSingleQuantileResult(IQMRange, ec, in.getFedMapping(), varID, false, true, vectorLength);
		}
		else {
			if(!retBuckets.isEmpty()) {
				// Search for values within bucket range where it as returned
				in.getFedMapping().mapParallel(varID, (range, data) -> {
					try {
						FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
							FederatedRequest.RequestType.EXEC_UDF,
							-1,
							new QuantilePickFEDInstruction.GetValuesInRanges(data.getVarID(), quantiles.length, (HashMap<Integer, ImmutablePair<Double, Double>>) retBuckets))).get();
						if(!response.isSuccessful())
							response.throwExceptionFromResponse();

						// Add results by row
						MatrixBlock tmp = (MatrixBlock) response.getData()[0];
						synchronized(out) {
							out.binaryOperationsInPlace(InstructionUtils.parseBinaryOperator("+"), tmp);
						}
						return null;
					}
					catch(Exception e) {
						throw new DMLRuntimeException(e);
					}
				});
			}

			ec.setMatrixOutput(output.getName(), out);
		}
	}

	private <T> void getSingleQuantileResult(T ret, ExecutionContext ec, FederationMap fedMap, long varID, boolean average, boolean isIQM, int vectorLength) {
		double result = 0.0;
		if(ret instanceof ImmutablePair) {
			// Search for values within bucket range
			List<Double> values = new ArrayList<>();
			fedMap.mapParallel(varID, (range, data) -> {
				try {
					FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
						-1,
						new QuantilePickFEDInstruction.GetValuesInRange(data.getVarID(), (ImmutablePair<Double, Double>) ret, isIQM))).get();
					if(!response.isSuccessful())
						response.throwExceptionFromResponse();
					values.add((double) response.getData()[0]);
					return null;
				}
				catch(Exception e) {
					throw new DMLRuntimeException(e);
				}
			});

			// Sum of 1 or 2 values
			result = values.stream().reduce(0.0, Double::sum);

		} else
			result = (Double) ret;

		result /= (average ? 2 : isIQM ? ((int) Math.round(vectorLength * 0.75) - (int) Math.round(vectorLength * 0.25)) : 1);

		ec.setScalarOutput(output.getName(), new DoubleObject(result));
	}

	public <T> T createHistogram(MatrixObject in, int vectorLength,  double globalMin, double globalMax, int numBuckets, int quantileIndex, boolean average) {
		FederationMap fedMap = in.getFedMapping();

		Map<ImmutablePair<Double, Double>, Integer> buckets = new LinkedHashMap<>();
		List<Map<ImmutablePair<Double, Double>, Integer>> hists = new ArrayList<>();
		List<Set<Double>> distincts = new ArrayList<>();

		double bucketRange = (globalMax-globalMin) / numBuckets;
		boolean isEvenNumRows = vectorLength % 2 == 0;

		// Create buckets according to min and max
		double tmpMin = globalMin, tmpMax = globalMax;
		for(int i = 0; i < numBuckets && tmpMin <= tmpMax; i++) {
			buckets.put(new ImmutablePair<>(tmpMin, tmpMin + bucketRange), 0);
			tmpMin += bucketRange;
		}

		// Create histograms
		long varID = FederationUtils.getNextFedDataID();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new QuantilePickFEDInstruction.GetHistogram(data.getVarID(), buckets, globalMax))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				Map<ImmutablePair<Double, Double>, Integer> rangeHist = (Map<ImmutablePair<Double, Double>, Integer>) response.getData()[0];
				hists.add(rangeHist);
				Set<Double> rangeDistinct = (Set<Double>) response.getData()[1];
				distincts.add(rangeDistinct);
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		// Merge results into one histogram
		for(ImmutablePair<Double, Double> bucket : buckets.keySet()) {
			int value = 0;
			for(Map<ImmutablePair<Double, Double>, Integer> hist : hists)
				value += hist.get(bucket);
			buckets.put(bucket, value);
		}

		if(quantileIndex == -1)
			return (T) buckets;

		// Find bucket with quantile
		ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> bucketWithIndex = getBucketWithIndex(buckets, quantileIndex, average, isEvenNumRows);

		// Check if can terminate
		Set<Double> distinctValues = distincts.stream().flatMap(Set::stream).collect(Collectors.toSet());
		if((distinctValues.size() == 1 && !average) || (distinctValues.size() == 2 && average))
			return (T) distinctValues.stream().reduce(0.0, (a, b) -> a + b);

		ImmutablePair<Double, Double> finalBucketWithQ = bucketWithIndex.right;
		List<Double> distinctInNewBucket = distinctValues.stream().filter( e -> e >= finalBucketWithQ.left && e <= finalBucketWithQ.right).collect(Collectors.toList());
		if((distinctInNewBucket.size() == 1 && !average) || (average && distinctInNewBucket.size() == 2))
			return (T) distinctInNewBucket.stream().reduce(0.0, (a, b) -> a + b);

		if(distinctValues.size() == 1 || (bucketWithIndex.middle == 1 && !average) || (bucketWithIndex.middle == 2 && isEvenNumRows && average) ||
			globalMin == globalMax)
			return (T) bucketWithIndex.right;

		int nextNumBuckets = bucketWithIndex.middle < 100 ? bucketWithIndex.middle * 2 : (int) Math.round(bucketWithIndex.middle / 2.0);

		// Add more bins to not stuck
		if(numBuckets == nextNumBuckets && globalMin == bucketWithIndex.right.left && globalMax == bucketWithIndex.right.right) {
			nextNumBuckets *= 2;
		}

		return createHistogram(in, vectorLength, bucketWithIndex.right.left, bucketWithIndex.right.right, nextNumBuckets, bucketWithIndex.left, average);
	}

	private ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> getBucketWithIndex(Map<ImmutablePair<Double, Double>, Integer> buckets, int quantileIndex, boolean average, boolean isEvenNumRows) {
		int sizeBeforeTmp = 0, sizeBefore = 0, bucketWithQSize = 0;
		ImmutablePair<Double, Double> bucketWithQ = null;
		for(Map.Entry<ImmutablePair<Double, Double>, Integer> range : buckets.entrySet()) {
			sizeBeforeTmp += range.getValue();
			if(quantileIndex <= sizeBeforeTmp && bucketWithQSize == 0) {
				bucketWithQ = range.getKey();
				bucketWithQSize = range.getValue();
				sizeBeforeTmp -= bucketWithQSize;
				sizeBefore = sizeBeforeTmp;

				if(!average || sizeBefore + bucketWithQSize >= quantileIndex + 1)
					break;
			} else if(quantileIndex + 1 <= sizeBeforeTmp + bucketWithQSize && isEvenNumRows && average) {
				// Add right bin that contains second index
				int bucket2Size = range.getValue();
				if (bucket2Size != 0) {
					bucketWithQ = new ImmutablePair<>(bucketWithQ.left, range.getKey().right);
					bucketWithQSize += bucket2Size;
					break;
				}
			}
		}
		quantileIndex = quantileIndex == 1 ? 1 : quantileIndex - sizeBefore;
		return new ImmutableTriple<>(quantileIndex, bucketWithQSize, bucketWithQ);
	}

	public static class GetHistogram extends FederatedUDF {
		private static final long serialVersionUID = 5413355823424777742L;
		private final Map<ImmutablePair<Double, Double>, Integer> _buckets;
		private final double _max;

		private GetHistogram(long input, Map<ImmutablePair<Double, Double>, Integer> buckets, double max) {
			super(new long[] {input});
			_buckets = buckets;
			_max = max;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] values = mb.getDenseBlockValues();
			boolean isWeighted  = mb.getNumColumns() == 2;

			Map<ImmutablePair<Double, Double>, Integer> hist = _buckets;
			Set<Double> distinct = new HashSet<>();

			for(int i = 0; i < values.length - (isWeighted ? 1 : 0); i += (isWeighted ? 2 : 1)) {
				double val = values[i];
				int weight = isWeighted ? (int) values[i+1] : 1;
				for (Map.Entry<ImmutablePair<Double, Double>, Integer> range : _buckets.entrySet()) {
					if((val >= range.getKey().left && val < range.getKey().right) || (val == _max && val == range.getKey().right)) {
						hist.put(range.getKey(), range.getValue() + weight);

						distinct.add(val);
					}
				}
			}

			Object[] ret = new Object[] {hist, distinct.size() < 3 ? distinct : new HashSet<>()};
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, ret);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class GetValuesInRanges extends FederatedUDF {
		private static final long serialVersionUID = 8663298932616139153L;
		private final int _numQuantiles;
		private final HashMap<Integer, ImmutablePair<Double, Double>> _ranges;

		private GetValuesInRanges(long input,int numQuantiles, HashMap<Integer, ImmutablePair<Double, Double>> ranges) {
			super(new long[] {input});
			_ranges = ranges;
			_numQuantiles = numQuantiles;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] values = mb.getDenseBlockValues();

			// FIXME rewrite - see binning encode
			MatrixBlock res = new MatrixBlock(_numQuantiles, 1, false);
			for(double val : values) {
				for(Map.Entry<Integer, ImmutablePair<Double, Double>> entry : _ranges.entrySet()) {
					if(entry.getValue().left <= val && val <= entry.getValue().right) {
						res.setValue(entry.getKey(), 0,val);
						break;
					}
				}
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, res);
		}

		@Override public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class MinMax extends FederatedUDF {
		private static final long serialVersionUID = -3906698363866500744L;

		private MinMax(long input) {
			super(new long[] {input});
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] ret = new double[]{mb.getNumColumns() == 2 ? mb.colMin().quickGetValue(0, 0) : mb.min(),
				mb.getNumColumns() == 2 ? mb.colMax().quickGetValue(0, 0) : mb.max(),
				mb.getNumColumns() == 2 ? mb.colSum().quickGetValue(0, 1) : 0,
				mb.getNumColumns() == 2 ? mb.sumWeightForQuantile() : 0};
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, ret);
		}

		@Override public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public void processColumnQPick(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMapping = in.getFedMapping();

		List <Object> res = new ArrayList<>();
		long varID = FederationUtils.getNextFedDataID();
		fedMapping.mapParallel(varID, (range, data) -> {
			FederatedResponse response;
			try {
				switch( _type )
				{
					case VALUEPICK:
						if(input2.isScalar()) {
							ScalarObject quantile = ec.getScalarInput(input2);
							response = data.executeFederatedOperation(
								new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,-1,
								new QuantilePickFEDInstruction.ValuePick(data.getVarID(), quantile))).get();
						}
						else {
							MatrixBlock quantiles = ec.getMatrixInput(input2.getName());
							response = data.executeFederatedOperation(
								new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,-1,
								new QuantilePickFEDInstruction.ValuePick(data.getVarID(), quantiles))).get();
						}
						break;
					case IQM:
						response = data
							.executeFederatedOperation(
								new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
								new QuantilePickFEDInstruction.ColIQM(data.getVarID()))).get();
						break;
					case MEDIAN:
						response = data
							.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
								new QuantilePickFEDInstruction.ColMedian(data.getVarID()))).get();
						break;
					default:
						throw new DMLRuntimeException("Unsupported qpick operation type: "+_type);
				}

				if(!response.isSuccessful())
					response.throwExceptionFromResponse();

				res.add(response.getData()[0]);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		assert res.size() == 1;

		if (input2 != null && input2.isMatrix())
			ec.releaseMatrixInput(input2.getName());

		if(output.isScalar())
			ec.setScalarOutput(output.getName(), new DoubleObject((double) res.get(0)));
		else
			ec.setMatrixOutput(output.getName(), (MatrixBlock) res.get(0));
	}

	private static class ValuePick extends FederatedUDF {

		private static final long serialVersionUID = -2594912886841345102L;
		private final MatrixBlock _quantiles;

		protected ValuePick(long input, ScalarObject quantile) {
			super(new long[] {input});
			_quantiles = new MatrixBlock(quantile.getDoubleValue());
		}

		protected ValuePick(long input, MatrixBlock quantiles) {
			super(new long[] {input});
			_quantiles = quantiles;
		}
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject)data[0]).acquireReadAndRelease();
			MatrixBlock picked;
			if (_quantiles.getLength() == 1) {
				return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS,
					new Object[] {mb.pickValue(_quantiles.getValue(0, 0))});
			}
			else {
				picked = mb.pickValues(_quantiles, new MatrixBlock());
				return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS,
					new Object[] {picked});
			}
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class GetValuesInRange extends FederatedUDF {
		private static final long serialVersionUID = 5413355823424777742L;
		private final ImmutablePair<Double, Double> _range;
		private final boolean _sumInRange;

		private GetValuesInRange(long input, ImmutablePair<Double, Double> range, boolean sumInRange) {
			super(new long[] {input});
			_range = range;
			_sumInRange = sumInRange;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] values = mb.getDenseBlockValues();

			double res = 0.0;
			int i = 0;

			// FIXME better search, e.g. sort in QSort and binary search
			for(double val : values) {
				// different conditions for IQM and simple QPICK
				if((!_sumInRange && _range.left <= val && val <= _range.right) ||
					(_sumInRange && _range.left < val && val <= _range.right))
					res += val;
				if(i++ > 2 && !_sumInRange)
					break;
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, res);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	private static class ColIQM extends FederatedUDF {

		private static final long serialVersionUID = 2223186699111957677L;

		protected ColIQM(long input) {
			super(new long[] {input});
		}
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject)data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS,
				new Object[] {mb.interQuartileMean()});
		}
		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	private static class ColMedian extends FederatedUDF {

		private static final long serialVersionUID = -2808597461054603816L;

		protected ColMedian(long input) {
			super(new long[] {input});
		}
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject)data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS,
				new Object[] {mb.median()});
		}
		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
