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
import java.util.stream.Collectors;
import java.util.stream.Stream;

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

	public <T> MatrixBlock getEquiHeightBins(ExecutionContext ec, int colID, double[] quantiles) {
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
		double globalMin = Double.MAX_VALUE, globalMax = Double.MIN_VALUE, vectorLength = inFrame.getNumColumns() == 2 ? 0 : inFrame.getNumRows();
		for(double[] values : minMax) {
			globalMin = Math.min(globalMin, values[0]);
			globalMax = Math.max(globalMax, values[1]);
		}

		// If multiple quantiles take first histogram and reuse bins, otherwise recursively get bin with result
		int numBuckets = 256; // (int) Math.round(in.getNumRows() / 2.0);

		T ret = createHistogram(in, (int) vectorLength, globalMin, globalMax, numBuckets, -1, false);

		// Compute and set results
		MatrixBlock quantileValues =  computeMultipleQuantiles(ec, in, (int[]) ret, quantiles, (int) vectorLength, varID, (globalMax-globalMin) / numBuckets, globalMin, _type, true);

		ec.removeVariable(String.valueOf(varID));

		// Add min to the result
		MatrixBlock res = new MatrixBlock(quantileValues.getNumRows() + 1, 1, false);
		res.set(0,0, globalMin);
		res.copy(1, quantileValues.getNumRows(), 0, 0,  quantileValues,false);

		return res;
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
			double finalVectorLength = vectorLength;
			quantiles = Arrays.stream(quantiles).map(val -> (int) Math.round(finalVectorLength * val)).toArray();
			computeMultipleQuantiles(ec, in, (int[]) ret, quantiles, (int) vectorLength, varID, (globalMax-globalMin) / numBuckets, globalMin, _type, false);
		}
		else
			getSingleQuantileResult(ret, ec, fedMap, varID, average, false, (int) vectorLength, null);
	}

	private <T> MatrixBlock computeMultipleQuantiles(ExecutionContext ec, MatrixObject in, int[] bucketsFrequencies, double[] quantiles,
		int vectorLength, long varID, double bucketRange, double min, OperationTypes type, boolean returnOutput) {
		MatrixBlock out = new MatrixBlock(quantiles.length, 1, false);
		ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>>[] bucketsWithIndex = new ImmutableTriple[quantiles.length];

		// Find bins with each quantile for first histogram
		int sizeBeforeTmp = 0, sizeBefore = 0, countFoundBins = 0;
		for(int j = 0; j < bucketsFrequencies.length; j++) {
			sizeBeforeTmp += bucketsFrequencies[j];

			for(int i = 0; i < quantiles.length; i++) {

				ImmutablePair<Double, Double> bucketWithQ;

				if(quantiles[i] > sizeBefore && quantiles[i] <= sizeBeforeTmp) {
					bucketWithQ = new ImmutablePair<>(min + (j * bucketRange), min + ((j+1) * bucketRange));
					bucketsWithIndex[i] = new ImmutableTriple<>(quantiles[i] == 1 ? 1 :
						(int) quantiles[i] - sizeBefore, bucketsFrequencies[j], bucketWithQ);
					countFoundBins++;
				}
			}

			sizeBefore = sizeBeforeTmp;
			if(countFoundBins == quantiles.length)
				break;
		}

		// Find each quantile bin recursively
		Map<Integer, T> retBuckets = new HashMap<>();

		double q25Left = 0, q25Right = 0, q75Left = 0, q75Right = 0;
		for(int i = 0; i < bucketsWithIndex.length; i++) {
			int nextNumBuckets = bucketsWithIndex[i].middle < 100 ? bucketsWithIndex[i].middle * 2 : (int) Math.round(bucketsWithIndex[i].middle / 2.0);
			T hist = createHistogram(in, vectorLength, bucketsWithIndex[i].right.left, bucketsWithIndex[i].right.right, nextNumBuckets, bucketsWithIndex[i].left, false);

			if(_type == OperationTypes.IQM) {
				q25Right = i == 0 ? hist instanceof ImmutablePair ?  ((ImmutablePair<Double, Double>)hist).right : (Double) hist : q25Right;
				q25Left = i == 0 ? hist instanceof ImmutablePair ?  ((ImmutablePair<Double, Double>)hist).left : (Double) hist : q25Left;
				q75Right = i == 1 ? hist instanceof ImmutablePair ? ((ImmutablePair<Double, Double>)hist).right : (Double) hist : q75Right;
				q75Left = i == 1 ? hist instanceof ImmutablePair ?  ((ImmutablePair<Double, Double>)hist).left : (Double) hist : q75Left;
			} else {
				if(hist instanceof ImmutablePair)
					retBuckets.put(i, hist); // set value if returned double instead of bin
				else
					out.set(i, 0, (Double) hist);
			}
		}

		if(type == OperationTypes.IQM) {
			ImmutablePair<Double, Double> IQMRange = new ImmutablePair<>(q25Right, q75Right);
			if(q25Right == q75Right)
				ec.setScalarOutput(output.getName(), new DoubleObject(q25Left));
			else
				getSingleQuantileResult(IQMRange, ec, in.getFedMapping(), varID, false, true, vectorLength, new ImmutablePair<>(q25Left, q75Left));
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
							out.binaryOperationsInPlace(InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString()), tmp);
						}
						return null;
					}
					catch(Exception e) {
						throw new DMLRuntimeException(e);
					}
				});
			}
			if(returnOutput)
				return out;
			else
				ec.setMatrixOutput(output.getName(), out);
		}
		return null;
	}

	private <T> void getSingleQuantileResult(T ret, ExecutionContext ec, FederationMap fedMap, long varID, boolean average, boolean isIQM, int vectorLength, ImmutablePair<Double, Double> iqmRange) {
		double result = 0.0, q25Part = 0, q25Val = 0, q75Val = 0, q75Part = 0;
		if(ret instanceof ImmutablePair) {
			// Search for values within bucket range
			List<Double> values = new ArrayList<>();
			List<double[]> iqmValues = new ArrayList<>();
			fedMap.mapParallel(varID, (range, data) -> {
				try {
					FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new QuantilePickFEDInstruction.GetValuesInRange(data.getVarID(), (ImmutablePair<Double, Double>) ret, isIQM, iqmRange))).get();
					if(!response.isSuccessful())
						response.throwExceptionFromResponse();
					if(isIQM)
						iqmValues.add((double[]) response.getData()[0]);
					else
						values.add((double) response.getData()[0]);
					return null;
				}
				catch(Exception e) {
					throw new DMLRuntimeException(e);
				}
			});


			if(isIQM) {
				for(double[] vals : iqmValues) {
					result += vals[0];
					q25Part += vals[1];
					q25Val += vals[2];
					q75Part += vals[3];
					q75Val += vals[4];
				}
				q25Part -= (0.25 * vectorLength);
				q75Part -= (0.75 * vectorLength);
			} else
				result = values.stream().reduce(0.0, Double::sum);

		} else
			result = (Double) ret;

		result = average ? result / 2 : (isIQM ? ((result + q25Part*q25Val - q75Part*q75Val) / (vectorLength * 0.5)) : result);

		ec.setScalarOutput(output.getName(), new DoubleObject(result));
	}

	public <T> T createHistogram(CacheableData<?> in, int vectorLength,  double globalMin, double globalMax, int numBuckets, int quantileIndex, boolean average) {
		FederationMap fedMap = in.getFedMapping();
		List<int[]> hists = new ArrayList<>();
		List<Set<Double>> distincts = new ArrayList<>();

		double bucketRange = (globalMax-globalMin) / numBuckets;
		boolean isEvenNumRows = vectorLength % 2 == 0;

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
		ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> bucketWithIndex = getBucketWithIndex(bucketsFrequencies, globalMin, quantileIndex, average, isEvenNumRows, bucketRange);

		// Check if can terminate
		Set<Double> distinctValues = distincts.stream().flatMap(Set::stream).collect(Collectors.toSet());

		if(distinctValues.size() > quantileIndex-1 && !average)
			return (T) distinctValues.stream().sorted().toArray()[quantileIndex > 0 ? quantileIndex-1 : 0];

		if(average && distinctValues.size() > quantileIndex) {
			Double[] distinctsSorted = distinctValues.stream().flatMap(Stream::of).sorted().toArray(Double[]::new);
			Double medianSum = Double.sum(distinctsSorted[quantileIndex-1], distinctsSorted[quantileIndex]);
			return (T) medianSum;
		}

		if((average && distinctValues.size() == 2) || (!average && distinctValues.size() == 1))
			return (T) distinctValues.stream().reduce(0.0, Double::sum);

		ImmutablePair<Double, Double> finalBucketWithQ = bucketWithIndex.right;
		List<Double> distinctInNewBucket = distinctValues.stream().filter( e -> e >= finalBucketWithQ.left && e <= finalBucketWithQ.right).collect(Collectors.toList());
		if((distinctInNewBucket.size() == 1 && !average) || (average && distinctInNewBucket.size() == 2))
			return (T) distinctInNewBucket.stream().reduce(0.0, Double::sum);

		if(!average) {
			Set<Double> distinctsSet = new HashSet<>(distinctInNewBucket);
			if(distinctsSet.size() == 1)
				return (T) distinctsSet.toArray()[0];
		}

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

	private ImmutableTriple<Integer, Integer, ImmutablePair<Double, Double>> getBucketWithIndex(int[] bucketFrequencies, double min, int quantileIndex, boolean average, boolean isEvenNumRows, double bucketRange) {
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

				if(!average || sizeBefore + bucketWithQSize >= quantileIndex + 1)
					break;
			} else if(quantileIndex + 1 <= sizeBeforeTmp + bucketWithQSize && isEvenNumRows && average) {
				// Add right bin that contains second index
				int bucket2Size = bucketFrequencies[i];
				if (bucket2Size != 0) {
					bucketWithQ = new ImmutablePair<>(bucketWithQ.left, tmpBinLeft + bucketRange);
					bucketWithQSize += bucket2Size;
					break;
				}
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
