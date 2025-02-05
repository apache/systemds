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

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.stats.TransformStatistics;

public class ColumnEncoderBin extends ColumnEncoder {
	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";
	private static final long serialVersionUID = 1917445005206076078L;

	public static final double SAMPLE_FRACTION = 0.1;
	public static final int MINIMUM_SAMPLE_SIZE = 1000;

	protected int _numBin = -1;
	private BinMethod _binMethod = BinMethod.EQUI_WIDTH;

	// frame transform-apply attributes
	// a) column bin boundaries
	private double[] _binMins = null;
	private double[] _binMaxs = null;
	// b) column min/max (for partial build)
	private double _colMins = -1f;
	private double _colMaxs = -1f;

	protected boolean containsNull = false;

	protected boolean checkedForNull = false;

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

	public int getNumBin() {
		return _numBin;
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
		if(method.equalsIgnoreCase(BinMethod.EQUI_WIDTH.toString()))
			_binMethod = BinMethod.EQUI_WIDTH;
		else if(method.equalsIgnoreCase(BinMethod.EQUI_HEIGHT.toString()))
			_binMethod = BinMethod.EQUI_HEIGHT;
		else if(method.equalsIgnoreCase(BinMethod.EQUI_HEIGHT_APPROX.toString()))
			_binMethod = BinMethod.EQUI_HEIGHT_APPROX;
		else
			throw new RuntimeException(method + " is invalid");
	}

	@Override
	public void build(CacheBlock<?> in) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(!isApplicable())
			return;
		else if(_binMethod == BinMethod.EQUI_WIDTH) {
			double[] pairMinMax = getMinMaxOfCol(in, _colID, 0, -1);
			computeBins(pairMinMax[0], pairMinMax[1]);
		}
		else if(_binMethod == BinMethod.EQUI_HEIGHT) {
			double[] sortedCol = prepareDataForEqualHeightBins(in, _colID, 0, -1);
			computeEqualHeightBins(sortedCol, false);
		}
		else if(_binMethod == BinMethod.EQUI_HEIGHT_APPROX){
			double[] vals = sampleDoubleColumn(in, _colID, SAMPLE_FRACTION, MINIMUM_SAMPLE_SIZE);
			Arrays.sort(vals);
			computeEqualHeightBins(vals, false);
		}

		if(in instanceof FrameBlock){
			final Array<?> c = ((FrameBlock )in).getColumn(_colID - 1);
			containsNull = c.containsNull();
			checkedForNull = true;
		}
		else {
			checkedForNull = true;
		}

		if(DMLScript.STATISTICS)
			TransformStatistics.incBinningBuildTime(System.nanoTime()-t0);
	}

	
	public void build(CacheBlock<?> in, double[] equiHeightMaxs) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(!isApplicable())
			return;
		else if(_binMethod == BinMethod.EQUI_WIDTH) {
			double[] pairMinMax = getMinMaxOfCol(in, _colID, 0, -1);
			computeBins(pairMinMax[0], pairMinMax[1]);
		}
		else if(_binMethod == BinMethod.EQUI_HEIGHT || _binMethod == BinMethod.EQUI_HEIGHT_APPROX) {
			computeEqualHeightBins(equiHeightMaxs, true);
		}

		if(DMLScript.STATISTICS)
			TransformStatistics.incBinningBuildTime(System.nanoTime()-t0);
	}

	protected double getCode(CacheBlock<?> in, int row){
		// find the right bucket for a single row
		if( _binMins.length == 0 || _binMaxs.length == 0 ) {
			LOG.warn("ColumnEncoderBin: applyValue without bucket boundaries, assign 1");
			return 1; //robustness in case of missing bins
		}
		// Returns NaN if value is missing, so can't be assigned a Bin
		double inVal = in.getDoubleNaN(row, _colID - 1);
		return getCodeIndex(inVal);
	}
	
	@Override
	protected final double[] getCodeCol(CacheBlock<?> in, int startInd, int endInd, double[] tmp) {
		final int endLength = endInd - startInd;
		final double[] codes = tmp != null && tmp.length == endLength ? tmp : new double[endLength];
		if (_binMins == null || _binMins.length == 0 || _binMaxs.length == 0) {
			LOG.warn("ColumnEncoderBin: applyValue without bucket boundaries, assign 1");
			Arrays.fill(codes, 0, endLength, 1.0);
			return codes;
		}

		if(in instanceof FrameBlock)
			getCodeColFrame((FrameBlock) in, startInd, endInd, codes);
		else{
			for (int i=startInd; i<endInd; i++) {
				double inVal = in.getDoubleNaN(i, _colID - 1);
				codes[i-startInd] = getCodeIndex(inVal);;
			}
		}
		return codes;
	}

	protected final void getCodeColFrame(FrameBlock in, int startInd, int endInd, double[] codes) {
		final Array<?> c = in.getColumn(_colID - 1);
		final double mi = _binMins[0];
		final double mx = _binMaxs[_binMaxs.length-1];
		if(!containsNull && checkedForNull)
			for(int i = startInd; i < endInd; i++)
				codes[i - startInd] = getCodeIndex(c.getAsDouble(i), mi, mx);
		else 
			for(int i = startInd; i < endInd; i++)
				codes[i - startInd] = getCodeIndex(c.getAsNaNDouble(i),mi, mx);
	}

	protected final double getCodeIndex(double inVal){
		return getCodeIndex(inVal, _binMins[0], _binMaxs[_binMaxs.length-1]);
	}

	protected final double getCodeIndex(double inVal, double min, double max){
		if(Double.isNaN(inVal))
			return Double.NaN;
		else if(_binMethod == BinMethod.EQUI_WIDTH)
			return getEqWidth(inVal, min, max);
		else // if (_binMethod == BinMethod.EQUI_HEIGHT || _binMethod == BinMethod.EQUI_HEIGHT_APPROX)
			return getCodeIndexEQHeight(inVal);
	}

	protected final double getEqWidth(double inVal, double min, double max) {
		if(max == min)
			return 1;
		return getEqWidthUnsafe(inVal, min, max);
	}

	protected final int getEqWidthUnsafe(double inVal){
		final double min = _binMins[0];
		final double max = _binMaxs[_binMaxs.length - 1];
		return getEqWidthUnsafe(inVal, min, max);
	}

	protected final int getEqWidthUnsafe(double inVal, double min, double max){
		final int code =  (int)(Math.ceil((inVal - min) / (max - min) * _numBin));
		return code > _numBin ? _numBin : code < 1 ? 1 : code;
	}


	private final double getCodeIndexEQHeight(double inVal){
		if(_binMaxs.length <= 10) 
			return getCodeIndexEQHeightSmall(inVal);
		else 
			return getCodeIndexEQHeightNormal(inVal);
	}

	private final double getCodeIndexEQHeightSmall(double inVal) {
		for(int i = 0; i < _binMaxs.length-1; i++)
			if(inVal <= _binMaxs[i])
				return i + 1;
		return _binMaxs.length;
	}
	
	private final double getCodeIndexEQHeightNormal(double inVal) {
		final int ix = Arrays.binarySearch(_binMaxs, inVal);
		if(ix < 0) // somewhere in between values
			// +2 because negative values are found from binary search.
			// plus 2 to correct for the absolute value of that.
			return Math.min(Math.abs(ix + 1) + 1, _binMaxs.length);
		else if(ix == 0) // If first bucket boundary add it there.
			return 1;
		else
			// precisely at boundaries default to lower bucket
			// This is done to avoid using an extra bucket for max value.
			return Math.min(ix + 1, _binMaxs.length);
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.BIN;
	}

	private static double[] getMinMaxOfCol(CacheBlock<?> in, int colID, int startRow, int blockSize) {
		// derive bin boundaries from min/max per column
		final int end = getEndIndex(in.getNumRows(), startRow, blockSize);
		if(in instanceof FrameBlock){
			FrameBlock inf = (FrameBlock) in;
			if(startRow == 0 && blockSize == -1)
				return inf.getColumn(colID -1).minMax();
			else
				return inf.getColumn(colID - 1).minMax(startRow, end);
		}

		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = startRow; i < end; i++) {
			final double inVal = in.getDoubleNaN(i, colID - 1);
			if(!Double.isNaN(inVal)){
				min = Math.min(min, inVal);
				max = Math.max(max, inVal);
			}
		}
		return new double[] {min, max};
	}

	private static double[] prepareDataForEqualHeightBins(CacheBlock<?> in, int colID, int startRow, int blockSize) {
		double[] vals = extractDoubleColumn(in, colID, startRow, blockSize);
		Arrays.sort(vals);
		return vals;
	}

	private static double[] extractDoubleColumn(CacheBlock<?> in, int colID, int startRow, int blockSize) {
		int endRow = getEndIndex(in.getNumRows(), startRow, blockSize);
		final int cid = colID -1;
		double[] vals = new double[endRow - startRow];
		if(in instanceof FrameBlock) {
			// FrameBlock optimization
			Array<?> a = ((FrameBlock) in).getColumn(cid);
			return a.extractDouble(vals, startRow, endRow);
		}
		else {
			for(int i = startRow; i < endRow; i++) {
				double inVal = in.getDoubleNaN(i, cid);
				// FIXME current NaN handling introduces 0s and thus
				// impacts the computation of bin boundaries
				if(Double.isNaN(inVal))
					continue;
				vals[i - startRow] = inVal;
			}
		}
		return vals;
	}

	private static double[] sampleDoubleColumn(CacheBlock<?> in, int colID, double sampleFraction, int minimum_sample_size){
		final int nRow = in.getNumRows();
		int elm =(int) Math.min( nRow, Math.max(minimum_sample_size, Math.ceil(nRow * sampleFraction)));
		double[] vals = new double[elm];
		Array<?> a = ((FrameBlock) in).getColumn(colID - 1);
		int s = DMLScript.SEED;
		Random r = s == -1 ? new Random() : new Random(s);
		for(int i = 0; i < elm; i++) {
			double inVal = a.getAsNaNDouble(r.nextInt(nRow));
			vals[i] = inVal;
		}
		return vals;

	}

	@Override
	public Callable<Object> getBuildTask(CacheBlock<?> in) {
		return new ColumnBinBuildTask(this, in);
	}

	@Override
	public Callable<Object> getPartialBuildTask(CacheBlock<?> in, int startRow, int blockSize,
			HashMap<Integer, Object> ret, int p) {
		return new BinPartialBuildTask(in, _colID, startRow, blockSize, _binMethod, ret);
	}

	@Override
	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret) {
		return new BinMergePartialBuildTask(this, ret);
	}

	public void computeBins(double min, double max) {
		if(min == max){
			_numBin = 1;
		}
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

	private void computeEqualHeightBins(double[] sortedCol, boolean doNotTakeQuantiles) {
		if(_binMins == null || _binMaxs == null) {
			_binMins = new double[_numBin];
			_binMaxs = new double[_numBin];
		}
		if(!doNotTakeQuantiles) {
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
		getSparseTask(CacheBlock<?> in, MatrixBlock out, int outputCol, int startRow, int blk) {
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

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append(": ");
		sb.append(_colID);
		sb.append(" --- Method: " + _binMethod + " num Bin: " + _numBin);
		sb.append("\n---- BinMin: " + Arrays.toString(_binMins));
		sb.append("\n---- BinMax: " + Arrays.toString(_binMaxs));
		return sb.toString();
	}

	public enum BinMethod {
		INVALID, EQUI_WIDTH, EQUI_HEIGHT, EQUI_HEIGHT_APPROX;

		@Override
		public String toString(){
			switch(this) {
				case EQUI_WIDTH: return "EQUI-WIDTH";
				case EQUI_HEIGHT: return "EQUI-HEIGHT";
				case EQUI_HEIGHT_APPROX: return "EQUI_HEIGHT_APPROX";
				default: throw new DMLRuntimeException("Invalid encoder type.");
			}
		}
	}

	private static class BinSparseApplyTask extends ColumnApplyTask<ColumnEncoderBin> {

		public BinSparseApplyTask(ColumnEncoderBin encoder, CacheBlock<?> input,
				MatrixBlock out, int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		private BinSparseApplyTask(ColumnEncoderBin encoder, CacheBlock<?> input, MatrixBlock out, int outputCol) {
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

		private final CacheBlock<?> _input;
		private final int _blockSize;
		private final int _startRow;
		private final int _colID;
		private final BinMethod _method;
		private final HashMap<Integer, Object> _partialData;

		// if a pool is passed the task may be split up into multiple smaller tasks.
		protected BinPartialBuildTask(CacheBlock<?> input, int colID, int startRow, 
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
			else if (_method == BinMethod.EQUI_HEIGHT || _method == BinMethod.EQUI_HEIGHT_APPROX) {
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
		private final CacheBlock<?> _input;

		protected ColumnBinBuildTask(ColumnEncoderBin encoder, CacheBlock<?> input) {
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
