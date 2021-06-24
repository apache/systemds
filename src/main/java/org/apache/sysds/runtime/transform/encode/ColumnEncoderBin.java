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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class ColumnEncoderBin extends ColumnEncoder {
	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";
	private static final long serialVersionUID = 1917445005206076078L;
	protected int _numBin = -1;

	// frame transform-apply attributes
	// a) column bin boundaries
	// TODO binMins is redundant and could be removed - necessary for correct fed results
	private double[] _binMins = null;
	private double[] _binMaxs = null;
	// b) column min/max (for partial build)
	private double _colMins = -1f;
	private double _colMaxs = -1f;

	public ColumnEncoderBin() {
		super(-1);
	}

	public ColumnEncoderBin(int colID, int numBin) {
		super(colID);
		_numBin = numBin;
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

	@Override
	public void build(FrameBlock in) {
		if(!isApplicable())
			return;
		double[] pairMinMax = getMinMaxOfCol(in, _colID, 0, -1);
		computeBins(pairMinMax[0], pairMinMax[1]);
	}

	private static double[] getMinMaxOfCol(FrameBlock in, int colID, int startRow, int blockSize){
		// derive bin boundaries from min/max per column
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = startRow; i < getEndIndex(in.getNumRows(), startRow, blockSize); i++) {
			double inVal = UtilFunctions.objectToDouble(in.getSchema()[colID - 1], in.get(i, colID - 1));
			min = Math.min(min, inVal);
			max = Math.max(max, inVal);
		}
		return new double[]{min, max};
	}


	@Override
	public Callable<Object> getBuildTask(FrameBlock in){
		return new ColumnBinBuildTask(this, in);
	}

	@Override
	public Callable<Object> getPartialBuildTask(FrameBlock in, int startRow, int blockSize, HashMap<Integer, Object> ret){
		return new BinPartialBuildTask(in, _colID, startRow, blockSize, ret);
	}

	@Override
	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret){
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

	public void prepareBuildPartial() {
		// ensure allocated min/max arrays
		_colMins = -1f;
		_colMaxs = -1f;
	}

	public void buildPartial(FrameBlock in) {
		if(!isApplicable())
			return;
		// derive bin boundaries from min/max per column
		double[] pairMinMax = getMinMaxOfCol(in, _colID, 0 ,-1);
		_colMins = pairMinMax[0];
		_colMaxs = pairMinMax[1];
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		for(int i = rowStart; i < getEndIndex(in.getNumRows(), rowStart, blk); i++) {
			double inVal = UtilFunctions.objectToDouble(in.getSchema()[_colID - 1], in.get(i, _colID - 1));
			int ix = Arrays.binarySearch(_binMaxs, inVal);
			int binID = ((ix < 0) ? Math.abs(ix + 1) : ix) + 1;
			out.quickSetValueThreadSafe(i, outputCol, binID);
		}
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		int end = (blk <= 0)? in.getNumRows(): in.getNumRows() < rowStart + blk ? in.getNumRows() : rowStart + blk;
		for(int i = rowStart; i < end; i++) {
			double inVal = in.quickGetValueThreadSafe(i, _colID - 1);
			int ix = Arrays.binarySearch(_binMaxs, inVal);
			int binID = ((ix < 0) ? Math.abs(ix + 1) : ix) + 1;
			out.quickSetValueThreadSafe(i, outputCol, binID);
		}
		return out;
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
	public FrameBlock getMetaData(FrameBlock meta) {
		// allocate frame if necessary
		meta.ensureAllocatedColumns(_binMaxs.length);

		// serialize the internal state into frame meta data
		meta.getColumnMetadata(_colID - 1).setNumDistinct(_numBin);
		for(int i = 0; i < _binMaxs.length; i++) {
			String sb = _binMins[i] +
					Lop.DATATYPE_PREFIX +
					_binMaxs[i];
			meta.set(i, _colID - 1, sb);
		}
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		if(meta == null || _binMaxs != null)
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

	private static class BinPartialBuildTask implements Callable<Object> {

		private final FrameBlock _input;
		private final int _blockSize;
		private final int _startRow;
		private final int _colID;
		private final HashMap<Integer, Object> _partialMinMax;

		// if a pool is passed the task may be split up into multiple smaller tasks.
		protected BinPartialBuildTask(FrameBlock input, int colID, int startRow, int blocksize,
									  HashMap<Integer, Object> partialMinMax){
			_input = input;
			_blockSize = blocksize;
			_colID = colID;
			_startRow = startRow;
			_partialMinMax = partialMinMax;
		}

		@Override
		public double[] call() throws Exception {
			_partialMinMax.put(_startRow, getMinMaxOfCol(_input, _colID, _startRow, _blockSize));
			return null;
		}
	}

	private static class BinMergePartialBuildTask implements Callable<Object>{
		private final HashMap<Integer, ?> _partialMaps;
		private final ColumnEncoderBin _encoder;

		private BinMergePartialBuildTask(ColumnEncoderBin encoderBin,
											HashMap<Integer, ?> partialMaps) {
			_partialMaps = partialMaps;
			_encoder = encoderBin;
		}

		@Override
		public Object call() throws Exception {
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			for(Object minMax: _partialMaps.values()){
				min = Math.min(min, ((double[]) minMax)[0]);
				max = Math.max(max, ((double[]) minMax)[1]);
			}
			_encoder.computeBins(min, max);
			return null;
		}
	}



	private static class ColumnBinBuildTask implements Callable<Object> {

		private final ColumnEncoderBin _encoder;
		private final FrameBlock _input;

		protected ColumnBinBuildTask(ColumnEncoderBin encoder, FrameBlock input) {
			_encoder = encoder;
			_input = input;
		}

		@Override
		public Void call() throws Exception {
			_encoder.build(_input);
			return null;
		}
	}

}
