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

package org.apache.sysds.runtime.transform.encode;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;
import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.stats.TransformStatistics;

public class ColumnEncoderBin extends ColumnEncoder {
	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";
	private static final long serialVersionUID = 1917445005206076078L;

	public int getNumBin() {
		return _numBin;
	}

	protected int _numBin = -1;
	private BinMethod _binMethod = BinMethod.EQUI_WIDTH;

	// frame transform-apply attributes
	// a) column bin boundaries
	private double[] _binMins = null;
	private double[] _binMaxs = null;
	// b) column min/max (for partial build)
	private double _colMins = -1f;
	private double _colMaxs = -1f;

	public ColumnEncoderBin() {
		super(-1);
	}

	public ColumnEncoderBin(int colID, int numBin, BinMethod binMethod)  {
		super(colID);
		_numBin = numBin;
		_binMethod = binMethod;
	}

	public ColumnEncoderBin(int colID, int numBin, double[] binMins, double[] binMaxs) {
		super(colID);
		_numBin = numBin;
		_binMins = binMins;
		_binMaxs = binMaxs;
	}

	public double getColMins() {
		return _colMins;
	}

	public double getColMaxs() {
		return _colMaxs;
	}

	public double[] getBinMins() {
		return _binMins;
	}

	public double[] getBinMaxs() {
		return _binMaxs;
	}

	public BinMethod getBinMethod() {
		return _binMethod;
	}

	public void setBinMethod(String method) {
		if (method.equalsIgnoreCase(BinMethod.EQUI_WIDTH.toString()))
			_binMethod = BinMethod.EQUI_WIDTH;
		if (method.equalsIgnoreCase(BinMethod.EQUI_HEIGHT.toString()))
			_binMethod = BinMethod.EQUI_HEIGHT;
	}

	@Override
	public void build(CacheBlock in) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(!isApplicable())
			return;
		if(_binMethod == BinMethod.EQUI_WIDTH) {
			double[] pairMinMax = getMinMaxOfCol(in, _colID, 0, -1);
			computeBins(pairMinMax[0], pairMinMax[1]);
		}
		else if(_binMethod == BinMethod.EQUI_HEIGHT) {
			double[] sortedCol = prepareDataForEqualHeightBins(in, _colID, 0, -1);
			computeEqualHeightBins(sortedCol, false);
		}

		if(DMLScript.STATISTICS)
			TransformStatistics.incBinningBuildTime(System.nanoTime()-t0);
	}

	public void build(CacheBlock in, double[] equiHeightMaxs) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(!isApplicable())
			return;
		if(_binMethod == BinMethod.EQUI_WIDTH) {
			double[] pairMinMax = getMinMaxOfCol(in, _colID, 0, -1);
			computeBins(pairMinMax[0], pairMinMax[1]);
		}
		else if(_binMethod == BinMethod.EQUI_HEIGHT) {
			computeEqualHeightBins(equiHeightMaxs, true);
		}

		if(DMLScript.STATISTICS)
			TransformStatistics.incBinningBuildTime(System.nanoTime()-t0);
	}

	protected double getCode(CacheBlock in, int row){
		// find the right bucket for a single row
		double bin = 0;
		if( _binMins.length == 0 || _binMaxs.length == 0 ) {
			LOG.warn("ColumnEncoderBin: applyValue without bucket boundaries, assign 1");
			return 1; //robustness in case of missing bins
		}
		// Returns NaN if value is missing, so can't be assigned a Bin
		double inVal = in.getDoubleNaN(row, _colID - 1);
		if (Double.isNaN(inVal) || inVal < _binMins[0] || inVal > _binMaxs[_binMaxs.length-1])
			return Double.NaN;
		if (_binMethod == BinMethod.EQUI_HEIGHT) {
			int ix = Arrays.binarySearch(_binMaxs, inVal);
			bin = ((ix < 0) ? Math.abs(ix + 1) : ix) + 1;
		}
		if (_binMethod == BinMethod.EQUI_WIDTH) {
			//TODO: Skip computing bin boundaries for equi-width
			double binWidth = (_binMaxs[_binMaxs.length - 1] - _binMins[0]) / _numBin;
			double code = Math.ceil((inVal - _binMins[0]) / binWidth);
			bin = (code == 0) ? code + 1 : code;
		}
		return bin;
	}
	
	@Override
	protected double[] getCodeCol(CacheBlock in, int startInd, int blkSize) {
		// find the right bucket for a block of rows
		int endInd = getEndIndex(in.getNumRows(), startInd, blkSize);
		double[] codes = new double[endInd-startInd];
		for (int i=startInd; i<endInd; i++) {
			if (_binMins == null || _binMins.length == 0 || _binMaxs.length == 0) {
				LOG.warn("ColumnEncoderBin: applyValue without bucket boundaries, assign 1");
				codes[i-startInd] = 1; //robustness in case of missing bins
				continue;
			}
			double inVal = in.getDoubleNaN(i, _colID - 1);
			if (Double.isNaN(inVal) || inVal < _binMins[0] || inVal > _binMaxs[_binMaxs.length-1]) {
				codes[i-startInd] = Double.NaN;
				continue;
			}
			if (_binMethod == BinMethod.EQUI_HEIGHT) {
				int ix = Arrays.binarySearch(_binMaxs, inVal);
				codes[i-startInd] = ((ix < 0) ? Math.abs(ix + 1) : ix) + 1;
			}
			if (_binMethod == BinMethod.EQUI_WIDTH) {
				//TODO: Skip computing bin boundaries for equi-width
				double binWidth = (_binMaxs[_binMaxs.length - 1] - _binMins[0]) / _numBin;
				double bin = Math.ceil((inVal - _binMins[0]) / binWidth);
				codes[i - startInd] = bin == 0 ? bin + 1 : bin;
			}
		}
		return codes;
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.BIN;
	}

	private static double[] getMinMaxOfCol(CacheBlock in, int colID, int startRow, int blockSize) {
		// derive bin boundaries from min/max per column
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = startRow; i < getEndIndex(in.getNumRows(), startRow, blockSize); i++) {
			double inVal = in.getDouble(i, colID - 1);
			if(Double.isNaN(inVal))
				continue;
			min = Math.min(min, inVal);
			max = Math.max(max, inVal);
		}
		return new double[] {min, max};
	}

	private static double[] prepareDataForEqualHeightBins(CacheBlock in, int colID, int startRow, int blockSize) {
		int endRow = getEndIndex(in.getNumRows(), startRow, blockSize);
		double[] vals = new double[endRow-startRow];
		for(int i = startRow; i < endRow; i++) {
			double inVal = in.getDouble(i, colID - 1);
			//FIXME current NaN handling introduces 0s and thus
			// impacts the computation of bin boundaries
			if(Double.isNaN(inVal))
				continue;
			vals[i-startRow] = inVal;
		}
		Arrays.sort(vals);
		return vals;
	}

	@Override
	public Callable<Object> getBuildTask(CacheBlock in) {
		return new ColumnBinBuildTask(this, in);
	}

	@Override
	public Callable<Object> getPartialBuildTask(CacheBlock in, int startRow, int blockSize,
			HashMap<Integer, Object> ret) {
		return new BinPartialBuildTask(in, _colID, startRow, blockSize, _binMethod, ret);
	}

	@Override
	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret) {
		return new BinMergePartialBuildTask(this, ret);
	}

	public void computeBins(double min, double max) {
		// ensure allocated internal transformation metadata
		if(_binMins == null || _binMaxs == null) {
			_binMins = new double[_numBin];
			_binMaxs = new double[_numBin];
		}
		for(int i = 0; i < _numBin; i++) {
			_binMins[i] = min + i * (max - min) / _numBin;
			_binMaxs[i] = min + (i + 1) * (max - min) / _numBin;
		}
	}

	private void computeEqualHeightBins(double[] sortedCol, boolean isSorted) {
		if(_binMins == null || _binMaxs == null) {
			_binMins = new double[_numBin];
			_binMaxs = new double[_numBin];
		}
		if(!isSorted) {
			int n = sortedCol.length;
			for(int i = 0; i < _numBin; i++) {
				double pos = n * (i + 1d) / _numBin;
				_binMaxs[i] = (pos % 1 == 0) ? // pos is integer
					sortedCol[(int) pos - 1] : sortedCol[(int) Math.floor(pos)];
			}
			_binMaxs[_numBin - 1] = sortedCol[n - 1];

		} else {
			System.arraycopy(sortedCol, 1, _binMaxs, 0, _numBin);
		}

		_binMins[0] = sortedCol[0];
		System.arraycopy(_binMaxs, 0, _binMins, 1, _numBin - 1);
	}

	public void prepareBuildPartial() {
		// ensure allocated min/max arrays
		_colMins = -1f;
		_colMaxs = -1f;
	}

	public void buildPartial(FrameBlock in) {
		if(!isApplicable())
			return;
		// derive bin boundaries from min/max per column
		double[] pairMinMax = getMinMaxOfCol(in, _colID, 0, -1);
		_colMins = pairMinMax[0];
		_colMaxs = pairMinMax[1];
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(CacheBlock in, MatrixBlock out, int outputCol, int startRow, int blk) {
		return new BinSparseApplyTask(this, in, out, outputCol);
	}

	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderBin) {
			ColumnEncoderBin otherBin = (ColumnEncoderBin) other;
			assert other._colID == _colID;
			// save the min, max as well as the number of bins for the column indexes
			MutableTriple<Integer, Double, Double> entry = new MutableTriple<>(_numBin, _binMins[0],
				_binMaxs[_binMaxs.length - 1]);
			// num bins will match
			entry.middle = Math.min(entry.middle, otherBin._binMins[0]);
			entry.right = Math.max(entry.right, otherBin._binMaxs[otherBin._binMaxs.length - 1]);

			// use the saved values to fill the arrays again
			_numBin = entry.left;
			_binMins = new double[_numBin];
			_binMaxs = new double[_numBin];

			double min = entry.middle;
			double max = entry.right;
			for(int j = 0; j < _numBin; j++) {
				_binMins[j] = min + j * (max - min) / _numBin;
				_binMaxs[j] = min + (j + 1) * (max - min) / _numBin;
			}
			return;
		}
		super.mergeAt(other);
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		meta.ensureAllocatedColumns(_binMaxs.length);
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		// allocate frame if necessary
		meta.ensureAllocatedColumns(_binMaxs.length);

		// serialize the internal state into frame meta data
		meta.getColumnMetadata(_colID - 1).setNumDistinct(_numBin);
		for(int i = 0; i < _binMaxs.length; i++) {
			String sb = _binMins[i] + Lop.DATATYPE_PREFIX + _binMaxs[i];
			meta.set(i, _colID - 1, sb);
		}
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		if(meta == null || _binMaxs != null || meta.getColumnMetadata()[_colID - 1].isDefault())
			return;
		// deserialize the frame meta data into internal state
		int nbins = (int) meta.getColumnMetadata()[_colID - 1].getNumDistinct();
		_binMins = new double[nbins];
		_binMaxs = new double[nbins];
		for(int i = 0; i < nbins; i++) {
			String[] tmp = meta.get(i, _colID - 1).toString().split(Lop.DATATYPE_PREFIX);
			_binMins[i] = Double.parseDouble(tmp[0]);
			_binMaxs[i] = Double.parseDouble(tmp[1]);
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);

		out.writeInt(_numBin);
		out.writeUTF(_binMethod.toString());
		out.writeBoolean(_binMaxs != null);
		if(_binMaxs != null) {
			for(int j = 0; j < _binMaxs.length; j++) {
				out.writeDouble(_binMaxs[j]);
				out.writeDouble(_binMins[j]);
			}
		}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_numBin = in.readInt();
		setBinMethod(in.readUTF());
		boolean minmax = in.readBoolean();
		_binMaxs = minmax ? new double[_numBin] : null;
		_binMins = minmax ? new double[_numBin] : null;
		if(!minmax)
			return;
		for(int j = 0; j < _binMaxs.length; j++) {
			_binMaxs[j] = in.readDouble();
			_binMins[j] = in.readDouble();
		}
	}

	public enum BinMethod {
		INVALID, EQUI_WIDTH, EQUI_HEIGHT
	}

	private static class BinSparseApplyTask extends ColumnApplyTask<ColumnEncoderBin> {

		public BinSparseApplyTask(ColumnEncoderBin encoder, CacheBlock input,
				MatrixBlock out, int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		private BinSparseApplyTask(ColumnEncoderBin encoder, CacheBlock input, MatrixBlock out, int outputCol) {
			super(encoder, input, out, outputCol);
		}

		public Object call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if(_out.getSparseBlock() == null)
				return null;
			_encoder.applySparse(_input, _out, _outputCol, _startRow, _blk);
			if(DMLScript.STATISTICS)
				TransformStatistics.incBinningApplyTime(System.nanoTime()-t0);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}
	}

	private static class BinPartialBuildTask implements Callable<Object> {

		private final CacheBlock _input;
		private final int _blockSize;
		private final int _startRow;
		private final int _colID;
		private final BinMethod _method;
		private final HashMap<Integer, Object> _partialData;

		// if a pool is passed the task may be split up into multiple smaller tasks.
		protected BinPartialBuildTask(CacheBlock input, int colID, int startRow, 
				int blocksize, BinMethod method, HashMap<Integer, Object> partialData) {
			_input = input;
			_blockSize = blocksize;
			_colID = colID;
			_startRow = startRow;
			_method = method;
			_partialData = partialData;
		}

		@Override
		public double[] call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if (_method == BinMethod.EQUI_WIDTH) {
				double[] minMax = getMinMaxOfCol(_input, _colID, _startRow, _blockSize);
				synchronized(_partialData) {
					_partialData.put(_startRow, minMax);
				}
			}
			if (_method == BinMethod.EQUI_HEIGHT) {
				double[] sortedVals = prepareDataForEqualHeightBins(_input, _colID, _startRow, _blockSize);
				synchronized(_partialData) {
					_partialData.put(_startRow, sortedVals);
				}
			}
			if (DMLScript.STATISTICS)
				TransformStatistics.incBinningBuildTime(System.nanoTime()-t0);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<Start row: " + _startRow + "; Block size: " + _blockSize + ">";
		}
	}

	private static class BinMergePartialBuildTask implements Callable<Object> {
		private final HashMap<Integer, ?> _partialMaps;
		private final ColumnEncoderBin _encoder;

		private BinMergePartialBuildTask(ColumnEncoderBin encoderBin, HashMap<Integer, ?> partialMaps) {
			_partialMaps = partialMaps;
			_encoder = encoderBin;
		}

		private double[] mergeKSortedArrays(double[][] arrs) {
			//PriorityQueue is heap in Java
			PriorityQueue<ArrayContainer> queue;
			queue = new PriorityQueue<>();
			int total=0;

			//add arrays to heap
			for(double[] arr : arrs) {
				queue.add(new ArrayContainer(arr, 0));
				total = total + arr.length;
			}
			int m=0;
			double[] result = new double[total];

			//while heap is not empty
			while(!queue.isEmpty()){
				ArrayContainer ac = queue.poll();
				result[m++]=ac.arr[ac.index];
				if(ac.index < ac.arr.length-1){
					queue.add(new ArrayContainer(ac.arr, ac.index+1));
				}
			}
			return result;
		}

		@Override
		public Object call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if (_encoder.getBinMethod() == BinMethod.EQUI_WIDTH) {
				double min = Double.POSITIVE_INFINITY;
				double max = Double.NEGATIVE_INFINITY;
				for(Object minMax : _partialMaps.values()) {
					min = Math.min(min, ((double[]) minMax)[0]);
					max = Math.max(max, ((double[]) minMax)[1]);
				}
				_encoder.computeBins(min, max);
			}

			if (_encoder.getBinMethod() == BinMethod.EQUI_HEIGHT) {
				double[][] allParts = new double[_partialMaps.size()][];
				int i = 0;
				for (Object arr: _partialMaps.values())
					allParts[i++] = (double[]) arr;

				// Heap-based merging of sorted partitions.
				// TODO: Derive bin boundaries from partial aggregates, avoiding
				// materializing the sorted arrays (e.g. federated quantile)
				double[] sortedRes = mergeKSortedArrays(allParts);
				_encoder.computeEqualHeightBins(sortedRes, false);
			}

			if(DMLScript.STATISTICS)
				TransformStatistics.incBinningBuildTime(System.nanoTime()-t0);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}
	}

	private static class ArrayContainer implements Comparable<ArrayContainer> {
		double[] arr;
		int index;

		public ArrayContainer(double[] arr, int index) {
			this.arr = arr;
			this.index = index;
		}

		@Override
		public int compareTo(ArrayContainer o) {
			return this.arr[this.index] < o.arr[o.index] ? -1 : 1;
		}
	}

	private static class ColumnBinBuildTask implements Callable<Object> {
		private final ColumnEncoderBin _encoder;
		private final CacheBlock _input;

		protected ColumnBinBuildTask(ColumnEncoderBin encoder, CacheBlock input) {
			_encoder = encoder;
			_input = input;
		}

		@Override
		public Void call() throws Exception {
			_encoder.build(_input);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}
	}
}
