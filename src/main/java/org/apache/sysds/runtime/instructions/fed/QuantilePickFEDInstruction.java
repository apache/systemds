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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.PickByCount.OperationTypes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.QuantilePickCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.QuantilePickSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

@SuppressWarnings("unchecked")
public class QuantilePickFEDInstruction extends BinaryFEDInstruction {

	private static final int NUM_BUCKETS = 256;

	private final OperationTypes _type;

	public QuantilePickFEDInstruction(Operator op, CPOperand in, CPOperand out, OperationTypes type, boolean inmem,
			String opcode, String istr) {
		this(op, in, null, out, type, inmem, opcode, istr);
	}

	public QuantilePickFEDInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
			boolean inmem, String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.QPick, op, in, in2, out, opcode, istr, fedOut);
		_type = type;
	}

	public QuantilePickFEDInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
		boolean inmem, String opcode, String istr) {
		this(op, in, in2, out, type, inmem, opcode, istr, FederatedOutput.NONE);
	}

	public static QuantilePickFEDInstruction parseInstruction(QuantilePickCPInstruction instr) {
		return new QuantilePickFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.output,
			instr.getOperationType(), instr.isInMem(), instr.getOpcode(), instr.getInstructionString());
	}

	public static QuantilePickFEDInstruction parseInstruction(QuantilePickSPInstruction instr) {
		return new QuantilePickFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.output,
				instr.getOperationType(), false, instr.getOpcode(), instr.getInstructionString());
	}

	public static QuantilePickFEDInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if ( !opcode.equalsIgnoreCase(Opcodes.QPICK.toString()) )
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantilePickCPInstruction: " + str);
		FederatedOutput fedOut = FederatedOutput.valueOf(parts[parts.length-1]);
		QuantilePickFEDInstruction inst = null;
		//instruction parsing
		if( parts.length == 5 ) {
			//instructions of length 5 originate from unary - mr-iqm
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.IQM;
			boolean inmem = false;
			inst = new QuantilePickFEDInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 6 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[2]);
			OperationTypes ptype = OperationTypes.valueOf(parts[3]);
			boolean inmem = Boolean.parseBoolean(parts[4]);
			inst = new QuantilePickFEDInstruction(null, in1, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 7 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.valueOf(parts[4]);
			boolean inmem = Boolean.parseBoolean(parts[5]);
			inst = new QuantilePickFEDInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		if ( inst != null )
			inst._fedOut = fedOut;
		return inst;
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(ec.getMatrixObject(input1).isFederated(FType.COL) || ec.getMatrixObject(input1).isFederated(FType.FULL))
			processColumnQPick(ec);
		else
			processRowQPick(ec);
	}

	public MatrixBlock getEquiHeightBins(ExecutionContext ec, int colID, double[] quantiles) {
		FrameObject inFrame = ec.getFrameObject(input1);
		FederationMap frameFedMap = inFrame.getFedMapping();

		// Create vector
		MatrixObject in = ExecutionContext.createMatrixObject(new MatrixBlock((int) inFrame.getNumRows(), 1, false));
		long varID = FederationUtils.getNextFedDataID();
		ec.setVariable(String.valueOf(varID), in);

		// modify map here
		List<FederatedRange> ranges = new ArrayList<>();
		FederationMap oldFedMap = frameFedMap.mapParallel(varID, (range, data) -> {
			try {
				int colIDWorker = colID;
				if(colID >= range.getBeginDims()[1] && colID < range.getEndDims()[1]) {
					if(range.getBeginDims()[1] > 1)
						colIDWorker = colID - (int) range.getBeginDims()[1];
					FederatedResponse response = data.executeFederatedOperation(
						new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
							new QuantilePickFEDInstruction.CreateMatrixFromFrame(data.getVarID(), varID, colIDWorker))).get();

					synchronized(ranges) {
						ranges.add(range);
					}
					if(!response.isSuccessful())
						response.throwExceptionFromResponse();
				}
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		//create one column federated object
		List<Pair<FederatedRange, FederatedData>> newFedMapPairs = new ArrayList<>();
		for(Pair<FederatedRange, FederatedData> mapPair : oldFedMap.getMap()) {
			for(FederatedRange r : ranges) {
				if(mapPair.getLeft().equals(r)) {
					newFedMapPairs.add(mapPair);
				}
			}
		}

		FederationMap newFedMap = new FederationMap(varID, newFedMapPairs, FType.COL);

		// construct a federated matrix with the encoded data
		in.getDataCharacteristics().setDimension(in.getNumRows(),1);
		in.setFedMapping(newFedMap);


		// Find min and max
		List<double[]> minMax = new ArrayList<>();
		newFedMap.mapParallel(varID, (range, data) -> {
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
		double globalMin = Double.MAX_VALUE, globalMax = Double.MIN_VALUE;
		int vectorLength = inFrame.getNumColumns() == 2 ? 0 : (int) inFrame.getNumRows();
		for(double[] values : minMax) {
			globalMin = Math.min(globalMin, values[0]);
			globalMax = Math.max(globalMax, values[1]);
		}

		// Equi-height bin boundaries: the caller (MultiReturnParameterizedBuiltinFEDInstruction) already
		// hands us raw 1-based ranks in [1..N] (numRows/numBin * (i+1)), not [0..1] probabilities, so use
		// them as ranks directly. Clamp to [1..N] to absorb float-rounding on the last edge.
		int[] ranks = Arrays.stream(quantiles).mapToInt(q -> Math.max(1, Math.min(vectorLength, (int) Math.round(q))))
			.toArray();
		Map<Integer, Double> rankToValue = pickMultipleRanks(in, ranks, vectorLength, varID, globalMin, globalMax);

		ec.removeVariable(String.valueOf(varID));

		// Result: [globalMin, v_r1, v_r2, ...] as a column vector.
		MatrixBlock res = new MatrixBlock(quantiles.length + 1, 1, false);
		res.set(0, 0, globalMin);
		for(int i = 0; i < quantiles.length; i++)
			res.set(i + 1, 0, rankToValue.get(ranks[i]));
		return res;
	}

	public void processRowQPick(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMap = in.getFedMapping();

		// Resolve requested probabilities.
		final double[] quantiles;
		if(input2 != null) {
			if(input2.isMatrix())
				quantiles = ec.getMatrixInput(input2).getDenseBlockValues();
			else if(input2.isScalar())
				quantiles = new double[] {ec.getScalarInput(input2).getDoubleValue()};
			else
				throw new DMLRuntimeException(
					"QuantilePickFEDInstruction: unsupported input2 data type " + input2.getDataType());
		}
		else if(_type == OperationTypes.MEDIAN)
			quantiles = new double[] {0.5};
		else if(_type == OperationTypes.IQM)
			quantiles = new double[] {0.25, 0.75};
		else
			throw new DMLRuntimeException("QuantilePickFEDInstruction: " + _type + " requires a probability input");

		if(input2 != null && input2.isMatrix())
			ec.releaseMatrixInput(input2.getName());

		// Fetch per-worker min/max/weights.
		long varID = FederationUtils.getNextFedDataID();
		List<double[]> minMax = new ArrayList<>();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new QuantilePickFEDInstruction.MinMax(data.getVarID()))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				minMax.add((double[]) response.getData()[0]);
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		double globalMin = Double.MAX_VALUE, globalMax = Double.MIN_VALUE;
		int vectorLength = in.getNumColumns() == 2 ? 0 : (int) in.getNumRows();
		double sumWeights = 0.0;
		for(double[] values : minMax) {
			globalMin = Math.min(globalMin, values[0]);
			globalMax = Math.max(globalMax, values[1]);
			if(in.getNumColumns() == 2)
				vectorLength += (int) values[2];
			sumWeights += values[3];
		}

		if(_type == OperationTypes.IQM) {
			computeIqm(ec, in, fedMap, varID, vectorLength, globalMin, globalMax);
			return;
		}

		// VALUEPICK / MEDIAN: R quantile type 7 — for each probability derive rank pair (lo, hi, g),
		// look up the deduplicated ranks once through the shared multi-rank pipeline, then interpolate.
		final long N = in.getNumColumns() == 2 ? Math.round(sumWeights) : vectorLength;
		final int[] los = new int[quantiles.length];
		final int[] his = new int[quantiles.length];
		final double[] gs = new double[quantiles.length];
		final Set<Integer> rankSet = new TreeSet<>();
		for(int i = 0; i < quantiles.length; i++) {
			final double[] r = MatrixBlock.computeQuantileRank(N, quantiles[i]);
			los[i] = (int) r[0];
			his[i] = (int) r[1];
			gs[i] = r[2];
			rankSet.add(los[i]);
			if(gs[i] > 0.0 && his[i] != los[i])
				rankSet.add(his[i]);
		}
		final int[] ranks = rankSet.stream().mapToInt(Integer::intValue).toArray();

		final Map<Integer, Double> rankToValue = pickMultipleRanks(in, ranks, vectorLength, varID, globalMin,
			globalMax);

		if(quantiles.length == 1) {
			ec.setScalarOutput(output.getName(),
				new DoubleObject((1.0 - gs[0]) * rankToValue.get(los[0]) + gs[0] * rankToValue.get(his[0])));
		}
		else {
			MatrixBlock out = new MatrixBlock(quantiles.length, 1, false);
			for(int i = 0; i < quantiles.length; i++)
				out.set(i, 0, (1.0 - gs[i]) * rankToValue.get(los[i]) + gs[i] * rankToValue.get(his[i]));
			ec.setMatrixOutput(output.getName(), out);
		}
	}

	// IQM is a trimmed weighted mean, not an R type-7 pick. Therefore, it uses raw ceil-based q25/q75 boundaries
	// and the closed-form boundary correction.
	private void computeIqm(ExecutionContext ec, MatrixObject in, FederationMap fedMap, long varID, int vectorLength,
		double globalMin, double globalMax) {
		final int q25Rank = (int) Math.ceil(0.25 * vectorLength);
		final int q75Rank = (int) Math.ceil(0.75 * vectorLength);

		final double bucketRange = (globalMax - globalMin) / NUM_BUCKETS;
		final int[] bucketsFrequencies = createHistogram(in, vectorLength, globalMin, globalMax, NUM_BUCKETS, -1);

		final int[] ranks = new int[] {q25Rank, q75Rank};
		final ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>>[] bucketsWithIndex = locateInitialBuckets(
			bucketsFrequencies, ranks, globalMin, bucketRange);

		double q25Left = 0, q25Right = 0, q75Left = 0, q75Right = 0;
		for(int i = 0; i < ranks.length; i++) {
			final Object hist = refineBucket(in, vectorLength, bucketsWithIndex[i]);
			final double left = hist instanceof ImmutablePair ? ((ImmutablePair<Double, Double>) hist).left : (Double) hist;
			final double right = hist instanceof ImmutablePair ? ((ImmutablePair<Double, Double>) hist).right : (Double) hist;
			if(i == 0) {
				q25Left = left;
				q25Right = right;
			}
			else {
				q75Left = left;
				q75Right = right;
			}
		}

		if(q25Right == q75Right) {
			ec.setScalarOutput(output.getName(), new DoubleObject(q25Left));
			return;
		}

		final ImmutablePair<Double, Double> iqmRange = new ImmutablePair<>(q25Right, q75Right);
		final ImmutablePair<Double, Double> bounds = new ImmutablePair<>(q25Left, q75Left);
		final List<double[]> perWorker = new ArrayList<>();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new QuantilePickFEDInstruction.GetValuesInRange(data.getVarID(), iqmRange, true, bounds)))
					.get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				perWorker.add((double[]) response.getData()[0]);
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		double sum = 0, q25Part = 0, q25Val = 0, q75Part = 0, q75Val = 0;
		for(double[] vals : perWorker) {
			sum += vals[0];
			q25Part += vals[1];
			q25Val += vals[2];
			q75Part += vals[3];
			q75Val += vals[4];
		}
		q25Part -= (0.25 * vectorLength);
		q75Part -= (0.75 * vectorLength);
		final double result = (sum + q25Part * q25Val - q75Part * q75Val) / (vectorLength * 0.5);
		ec.setScalarOutput(output.getName(), new DoubleObject(result));
	}

	// Look up the value at each requested (deduplicated, sorted) 1-based rank. Builds the coarse histogram once and
	// refines per rank.
	private Map<Integer, Double> pickMultipleRanks(MatrixObject in, int[] ranks, int vectorLength, long varID,
		double globalMin, double globalMax) {
		final Map<Integer, Double> result = new HashMap<>();
		if(ranks.length == 0)
			return result;

		// Single rank: skip the shared-histogram scaffolding and let createHistogram do its own initial build + refine.
		if(ranks.length == 1) {
			final Object hist = createHistogram(in, vectorLength, globalMin, globalMax, NUM_BUCKETS, ranks[0]);
			if(hist instanceof ImmutablePair)
				result.put(ranks[0],
					fetchValueInRange(in.getFedMapping(), varID, (ImmutablePair<Double, Double>) hist));
			else
				result.put(ranks[0], (Double) hist);
			return result;
		}

		final double bucketRange = (globalMax - globalMin) / NUM_BUCKETS;
		final int[] bucketsFrequencies = createHistogram(in, vectorLength, globalMin, globalMax, NUM_BUCKETS, -1);
		final ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>>[] bucketsWithIndex = locateInitialBuckets(
			bucketsFrequencies, ranks, globalMin, bucketRange);

		final HashMap<Integer, ImmutablePair<Double, Double>> retBuckets = new HashMap<>();
		for(int i = 0; i < ranks.length; i++) {
			final Object hist = refineBucket(in, vectorLength, bucketsWithIndex[i]);
			if(hist instanceof ImmutablePair)
				retBuckets.put(i, (ImmutablePair<Double, Double>) hist);
			else
				result.put(ranks[i], (Double) hist);
		}

		if(!retBuckets.isEmpty()) {
			final MatrixBlock resolved = new MatrixBlock(ranks.length, 1, false);
			in.getFedMapping().mapParallel(varID, (range, data) -> {
				try {
					FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
						FederatedRequest.RequestType.EXEC_UDF, -1,
						new QuantilePickFEDInstruction.GetValuesInRanges(data.getVarID(), ranks.length, retBuckets)))
						.get();
					if(!response.isSuccessful())
						response.throwExceptionFromResponse();
					MatrixBlock tmp = (MatrixBlock) response.getData()[0];
					synchronized(resolved) {
						resolved.binaryOperationsInPlace(
							InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString()), tmp);
					}
					return null;
				}
				catch(Exception e) {
					throw new DMLRuntimeException(e);
				}
			});
			for(Map.Entry<Integer, ImmutablePair<Double, Double>> entry : retBuckets.entrySet())
				result.put(ranks[entry.getKey()], resolved.get(entry.getKey(), 0));
		}

		return result;
	}

	// Refine a coarse-histogram bucket into a finer sub-histogram covering just that bucket's range, and recurse
	// into it for the given target rank. Returns either the final value (Double) or the bucket range.
	private Object refineBucket(MatrixObject in, int vectorLength,
		ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> bucketWithIndex) {
		final int nextNumBuckets = bucketWithIndex.middle < 100 ? bucketWithIndex.middle *
			2 : (int) Math.round(bucketWithIndex.middle / 2.0);
		return createHistogram(in, vectorLength, bucketWithIndex.right.left, bucketWithIndex.right.right,
			nextNumBuckets, bucketWithIndex.left);
	}

	// Scan the coarse histogram once and record, for each target rank, the bucket range containing it plus the
	// rank's offset within that bucket.
	// Triple layout per rank i: left = rank offset within the bucket (1-based, i.e. how many entries into the
	// bucket the target sits), middle = bucket frequency, right = (bucketMin, bucketMax) sub-range to recurse into.
	private static ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>>[] locateInitialBuckets(
		int[] bucketsFrequencies, int[] ranks, double globalMin, double bucketRange) {
		final ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>>[] bucketsWithIndex = new ImmutableTriple[ranks.length];
		int sizeBeforeTmp = 0, sizeBefore = 0, countFoundBins = 0;
		for(int j = 0; j < bucketsFrequencies.length; j++) {
			sizeBeforeTmp += bucketsFrequencies[j];
			for(int i = 0; i < ranks.length; i++) {
				if(bucketsWithIndex[i] == null && ranks[i] > sizeBefore && ranks[i] <= sizeBeforeTmp) {
					ImmutablePair<Double, Double> bucketWithR = new ImmutablePair<>(globalMin + (j * bucketRange),
						globalMin + ((j + 1) * bucketRange));
					bucketsWithIndex[i] = new ImmutableTriple<>(ranks[i] == 1 ? 1 : ranks[i] - sizeBefore,
						bucketsFrequencies[j], bucketWithR);
					countFoundBins++;
				}
			}
			sizeBefore = sizeBeforeTmp;
			if(countFoundBins == ranks.length)
				break;
		}
		return bucketsWithIndex;
	}

	private double fetchValueInRange(FederationMap fedMap, long varID, ImmutablePair<Double, Double> range) {
		final List<Double> values = new ArrayList<>();
		fedMap.mapParallel(varID, (r, data) -> {
			try {
				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new QuantilePickFEDInstruction.GetValuesInRange(data.getVarID(), range, false, null)))
					.get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				values.add((double) response.getData()[0]);
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});
		return values.stream().reduce(0.0, Double::sum);
	}

	public <T> T createHistogram(CacheableData<?> in, int vectorLength, double globalMin, double globalMax,
		int numBuckets, int quantileIndex) {
		FederationMap fedMap = in.getFedMapping();
		List<int[]> hists = new ArrayList<>();
		List<Set<Double>> distincts = new ArrayList<>();

		double bucketRange = (globalMax - globalMin) / numBuckets;

		// Create histograms
		long varID = FederationUtils.getNextFedDataID();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new QuantilePickFEDInstruction.GetHistogram(data.getVarID(), globalMin, globalMax, bucketRange, numBuckets))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				int[] rangeHist = (int[]) response.getData()[0];
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
		int[] bucketsFrequencies = new int[numBuckets];
		for(int[] hist : hists)
			for(int i = 0; i < hist.length; i++)
				bucketsFrequencies[i] += hist[i];

		if(quantileIndex == -1)
			return (T) bucketsFrequencies;

		// Find bucket with quantile
		ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> bucketWithIndex = getBucketWithIndex(
			bucketsFrequencies, globalMin, quantileIndex, bucketRange);

		// Check if we can terminate early using merged per-worker distincts.
		Set<Double> distinctValues = distincts.stream().flatMap(Set::stream).collect(Collectors.toSet());
		if(distinctValues.size() > quantileIndex - 1)
			return (T) distinctValues.stream().sorted().toArray()[quantileIndex > 0 ? quantileIndex - 1 : 0];

		if(distinctValues.size() == 1)
			return (T) distinctValues.stream().reduce(0.0, Double::sum);

		ImmutablePair<Double, Double> finalBucketWithQ = bucketWithIndex.right;
		List<Double> distinctInNewBucket = distinctValues.stream()
			.filter(e -> e >= finalBucketWithQ.left && e <= finalBucketWithQ.right).collect(Collectors.toList());
		if(distinctInNewBucket.size() == 1)
			return (T) distinctInNewBucket.get(0);

		Set<Double> distinctsSet = new HashSet<>(distinctInNewBucket);
		if(distinctsSet.size() == 1)
			return (T) distinctsSet.toArray()[0];

		if(bucketWithIndex.middle == 1 || globalMin == globalMax)
			return (T) bucketWithIndex.right;

		int nextNumBuckets = bucketWithIndex.middle < 100 ? bucketWithIndex.middle *
			2 : (int) Math.round(bucketWithIndex.middle / 2.0);

		// Add more bins to not stuck
		if(numBuckets == nextNumBuckets && globalMin == bucketWithIndex.right.left &&
			globalMax == bucketWithIndex.right.right) {
			nextNumBuckets *= 2;
		}

		return createHistogram(in, vectorLength, bucketWithIndex.right.left, bucketWithIndex.right.right,
			nextNumBuckets, bucketWithIndex.left);
	}

	// Locate the single bucket containing quantileIndex in a refinement histogram (called during recursion).
	// Triple layout: left = rank offset within the bucket (1-based), middle = bucket frequency,
	// right = (bucketMin, bucketMax) sub-range to recurse into next.
	private ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> getBucketWithIndex(int[] bucketFrequencies,
		double min, int quantileIndex, double bucketRange) {
		int sizeBeforeTmp = 0, sizeBefore = 0, bucketWithQSize = 0;
		ImmutablePair<Double, Double> bucketWithQ = null;

		double tmpBinLeft = min;
		for(int i = 0; i < bucketFrequencies.length; i++) {
			sizeBeforeTmp += bucketFrequencies[i];
			if(quantileIndex <= sizeBeforeTmp && bucketWithQSize == 0) {
				bucketWithQ = new ImmutablePair<>(tmpBinLeft, tmpBinLeft + bucketRange);
				bucketWithQSize = bucketFrequencies[i];
				sizeBeforeTmp -= bucketWithQSize;
				sizeBefore = sizeBeforeTmp;
				break;
			}
			tmpBinLeft += bucketRange;
		}
		quantileIndex = quantileIndex == 1 ? 1 : quantileIndex - sizeBefore;
		return new ImmutableTriple<>(quantileIndex, bucketWithQSize, bucketWithQ);
	}

	public static class CreateMatrixFromFrame extends FederatedUDF {
		private static final long serialVersionUID = -6569370318237863595L;
		private final long _outputID;
		private final int _id;

		public CreateMatrixFromFrame(long input, long output, int id) {
			super(new long[] {input});
			_outputID = output;
			_id = id;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FrameBlock fb = ((FrameObject) data[0]).acquireReadAndRelease();

			double[] colData = ArrayUtils.toPrimitive(Arrays.stream((Object[]) fb.getColumnData(_id)).map(e -> Double.valueOf(String.valueOf(e))).toArray(Double[] :: new));

			MatrixBlock mbout = new MatrixBlock(fb.getNumRows(), 1, colData);

			// create output matrix object
			MatrixObject mo = ExecutionContext.createMatrixObject(mbout);

			// add it to the list of variables
			ec.setVariable(String.valueOf(_outputID), mo);

			// return id handle
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS_EMPTY);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class GetHistogram extends FederatedUDF {
		private static final long serialVersionUID = 5413355823424777742L;
		private final double _max;
		private final double _min;
		private final double _range;
		private final int _numBuckets;

		private GetHistogram(long input, double min, double max, double range, int numBuckets) {
			super(new long[] {input});
			_max = max;
			_min = min;
			_range = range;
			_numBuckets = numBuckets;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] values = mb.getDenseBlockValues();
			boolean isWeighted  = mb.getNumColumns() == 2;

			Set<Double> distinct = new HashSet<>();

			int[] frequencies = new int[_numBuckets];

			// binning
			for(int i = 0; i < values.length - (isWeighted ? 1 : 0); i += (isWeighted ? 2 : 1)) {
				double val = values[i];
				int weight = isWeighted ? (int) values[i+1] : 1;
				int index = (int) (Math.ceil((val - _min) / _range));
				index = index == 0 ? 0 : index - 1;
				if (val >= _min && val <= _max) {
					frequencies[index] += weight;
					distinct.add(val);
				}
			}

			Object[] ret = new Object[] {frequencies, distinct.size() < 3 ? distinct : new HashSet<>()};
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

			MatrixBlock res = new MatrixBlock(_numQuantiles, 1, false);
			for(double val : values) {
				for(Map.Entry<Integer, ImmutablePair<Double, Double>> entry : _ranges.entrySet()) {
					// Find value within computed bin
					if(entry.getValue().left <= val && val <= entry.getValue().right) {
						res.set(entry.getKey(), 0,val);
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
			double[] ret = new double[]{mb.getNumColumns() == 2 ? mb.colMin().get(0, 0) : mb.min(),
				mb.getNumColumns() == 2 ? mb.colMax().get(0, 0) : mb.max(),
				mb.getNumColumns() == 2 ? mb.colSum().get(0, 1) : 0,
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
						// MEDIAN is VALUEPICK at p = 0.5
						response = data
							.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
								new QuantilePickFEDInstruction.ValuePick(data.getVarID(), new DoubleObject(0.5))))
							.get();
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
					new Object[] {mb.pickValue(_quantiles.get(0, 0))});
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
		private final ImmutablePair<Double, Double> _iqmRange;
		private final boolean _sumInRange;

		private GetValuesInRange(long input, ImmutablePair<Double, Double> range, boolean sumInRange, ImmutablePair<Double, Double> iqmRange) {
			super(new long[] {input});
			_range = range;
			_sumInRange = sumInRange;
			_iqmRange = iqmRange;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] values = mb.getDenseBlockValues();

			boolean isWeighted  = mb.getNumColumns() == 2;

			double res = 0.0;
			int counter = 0;

			double q25Part = 0, q25Val = 0, q75Val = 0, q75Part = 0;
			for(int i = 0; i < values.length - (isWeighted ? 1 : 0); i += (isWeighted ? 2 : 1)) {
				// get value within computed bin
				// different conditions for IQM and simple QPICK
				double val = values[i];
				int weight = isWeighted ? (int) values[i+1] : 1;

				if(_iqmRange != null && val <= _iqmRange.left) {
					q25Part += weight;
				}

				if(_iqmRange != null && val >= _iqmRange.left && val <= _range.left) {
					q25Val = val;
				}
				else if(_iqmRange != null && val <= _iqmRange.right && val >= _range.right)
					q75Val = val;

				if((!_sumInRange && _range.left <= val && val <= _range.right) ||
					(_sumInRange && _range.left < val && val <= _range.right)) {
					res += (val * (!_sumInRange && weight > 1 ? 2 : weight));
					counter += weight;
				}

				if(_iqmRange != null && val <= _range.right)
					q75Part += weight;

				if(!_sumInRange && counter > 2)
					break;
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS,!_sumInRange ? res : new double[]{res, q25Part, q25Val, q75Part, q75Val});
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
}
