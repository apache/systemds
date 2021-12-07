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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.ImmutablePair;
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
			//TODO this should be refactored to use pickvaluecount lops
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
		if(ec.getMatrixObject(input1).isFederated(FederationMap.FType.ROW))
			processRowQPick(ec);
		else
			processColumnQPick(ec);
	}

	public void processRowQPick(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMap = in.getFedMapping();

		boolean isEvenNumRows = in.getNumRows() % 2 == 0;
		boolean average = _type == OperationTypes.MEDIAN;

		ScalarObject quantile = input2 != null && input2.isScalar() ? ec.getScalarInput(input2) : average ? new DoubleObject(0.5) : null;

		// Find global min and max
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

		double globalMin = minMax.stream().mapToDouble(v -> v[0]).min().getAsDouble();
		double globalMax = minMax.stream().mapToDouble(v -> v[1]).max().getAsDouble();

		// FIXME appropriate number of buckets if not from min-max, moreover can reduce depth of recursion
		int numBuckets = (int) Math.round(in.getNumRows() / 2.0);
		int quantileIndex = (int) Math.round(in.getNumRows() * quantile.getDoubleValue());

		ImmutablePair<Double, Double> rangeWithQuantile = createHistogram(in, globalMin, globalMax, numBuckets, quantileIndex, average);

		// Search for values within bucket range
		List<double[]> values = new ArrayList<>();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new QuantilePickFEDInstruction.GetValuesInRange(data.getVarID(), rangeWithQuantile,
						new double[in.getNumRows() % 2 == 0 ? 2 : 1]))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				double[] rangeHist = (double[]) response.getData()[0];
				values.add(rangeHist);
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		// Sum of 1 or 2 values
		double result = 0.0;
		for(double[] val : values) {
			result += Arrays.stream(val).sum();
		}
		result /= isEvenNumRows && average ? 2 : 1;

		ec.setScalarOutput(output.getName(), new DoubleObject(result));
	}

	public ImmutablePair<Double, Double> createHistogram(MatrixObject in, double globalMin, double globalMax, int numBuckets, int quantileIndex, boolean average) {
		FederationMap fedMap = in.getFedMapping();

		Map<ImmutablePair<Double, Double>, Integer> buckets = new LinkedHashMap<>();
		List<Map<ImmutablePair<Double, Double>, Integer>> hists = new ArrayList<>();

		double bucketRange = (globalMax-globalMin) / numBuckets;
		boolean isEvenNumRows = in.getNumRows() % 2 == 0;

		// Create buckets according to min and max
		double tmpMin = globalMin, tmpMax = globalMax;
		for(int i = 0; i < numBuckets && tmpMin <= tmpMax; i++) {
			buckets.put(new ImmutablePair(tmpMin, tmpMin + bucketRange), 0);
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
				return null;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		});

		// Merge results into one histogram
		for(ImmutablePair<Double, Double> bucket : buckets.keySet()) {
			int value = 0;
			for(Map<ImmutablePair<Double, Double>, Integer> hist : hists) {
				value += hist.get(bucket);
			}
			buckets.put(bucket, value);
		}

		// Find bucket with quantile
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

		if((bucketWithQSize == 1 && !average) || (bucketWithQSize == 2 && isEvenNumRows && average) ||
			globalMin == globalMax)
			return bucketWithQ;

		// FIXME New number of buckets
//		if(globalMin == bucketWithQ.left && globalMax == bucketWithQ.right) {
//			numBuckets = numBuckets * 2;
//		} else
		numBuckets = bucketWithQSize < 100 ? bucketWithQSize * 2 : (int) Math.round(bucketWithQSize / 2.0);
		quantileIndex = quantileIndex == 1 ? 1 : quantileIndex - sizeBefore;
		return createHistogram(in, bucketWithQ.left, bucketWithQ.right, numBuckets, quantileIndex, average);
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

			Map<ImmutablePair<Double, Double>, Integer> hist = _buckets;
			for(double val : values) {
				for (Map.Entry<ImmutablePair<Double, Double>, Integer> range : _buckets.entrySet()) {
					if((val >= range.getKey().left && val < range.getKey().right) || (val == _max && val == range.getKey().right)) {
						hist.put(range.getKey(), range.getValue() + 1);
					}
				}
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, hist);
		}

		@Override public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class GetValuesInRange extends FederatedUDF {
		private static final long serialVersionUID = 5413355823424777742L;
		private final ImmutablePair<Double, Double> _range;
		private final double[] _out;

		private GetValuesInRange(long input, ImmutablePair<Double, Double> range, double[] out) {
			super(new long[] {input});
			_range = range;
			_out = out;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			double[] values = mb.getDenseBlockValues();

			int i = 0;
			for(double val : values) {
				if(_range.left <= val && val <= _range.right)
					_out[i++] = val;
				if(i > 1)
					break;
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, _out);
		}

		@Override public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	private static int binarySearch(double[] array, double elem) {
		int l = 0;
		int r = array.length - 1;
		while (l <= r) {
			int m = (int) Math.floor((l+r) / 2.0);

			if (array[m] < elem)
				l = m + 1;
			else if (array[m] > elem)
				r = m - 1;
			else
				return m;
		}
		return -1;
	}

	public static class MinMax extends FederatedUDF {
		private static final long serialVersionUID = -3906698363866500744L;

		private MinMax(long input) {
			super(new long[] {input});
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new double[]{mb.min(), mb.max()});
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
								new QuantilePickFEDInstruction.IQM(data.getVarID()))).get();
						break;
					case MEDIAN:
						response = data
							.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
								new QuantilePickFEDInstruction.Median(data.getVarID()))).get();
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

	private static class IQM extends FederatedUDF {

		private static final long serialVersionUID = 2223186699111957677L;

		protected IQM(long input) {
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

	private static class Median extends FederatedUDF {

		private static final long serialVersionUID = -2808597461054603816L;

		protected Median(long input) {
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
