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

import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class ColumnEncoderBin extends ColumnEncoder
{
	private static final long serialVersionUID = 1917445005206076078L;

	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";

	protected int _numBin = -1;

	//frame transform-apply attributes
	// a) column bin boundaries
	//TODO binMins is redundant and could be removed - necessary for correct fed results
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
		if ( !isApplicable() )
			return;

		// derive bin boundaries from min/max per column
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for( int i=0; i<in.getNumRows(); i++ ) {
			double inVal = UtilFunctions.objectToDouble(
				in.getSchema()[_colID-1], in.get(i, _colID-1));
			min = Math.min(min, inVal);
			max = Math.max(max, inVal);
		}
		computeBins(min, max);
	}
	
	public void computeBins(double min, double max) {
		// ensure allocated internal transformation metadata
		if( _binMins == null || _binMaxs == null ) {
			_binMins = new double[_numBin];
			_binMaxs = new double[_numBin];
		}
		for(int i=0; i<_numBin; i++) {
			_binMins[i] = min + i*(max-min)/_numBin;
			_binMaxs[i] = min + (i+1)*(max-min)/_numBin;
		}
	}
	
	public void prepareBuildPartial() {
		//ensure allocated min/max arrays
		_colMins = -1f;
		_colMaxs = -1f;
	}
	
	public void buildPartial(FrameBlock in) {
		if ( !isApplicable() )
			return;
		// derive bin boundaries from min/max per column
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for( int i=0; i<in.getNumRows(); i++ ) {
			double inVal = UtilFunctions.objectToDouble(
				in.getSchema()[_colID-1], in.get(i, _colID-1));
			min = Math.min(min, inVal);
			max = Math.max(max, inVal);
		}
		_colMins = min;
		_colMaxs = max;
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol) {
		for( int i=0; i<in.getNumRows(); i++ ) {
			double inVal = UtilFunctions.objectToDouble(
				in.getSchema()[_colID-1], in.get(i, _colID-1));
			int ix = Arrays.binarySearch(_binMaxs, inVal);
			int binID = ((ix < 0) ? Math.abs(ix+1) : ix) + 1;
			out.quickSetValue(i, outputCol, binID);
		}
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		for( int i=0; i<in.getNumRows(); i++ ) {
			double inVal = in.quickGetValue(i, _colID-1);
			int ix = Arrays.binarySearch(_binMaxs, inVal);
			int binID = ((ix < 0) ? Math.abs(ix+1) : ix) + 1;
			out.quickSetValue(i, outputCol, binID);
		}
		return out;
	}

	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderBin) {
			ColumnEncoderBin otherBin = (ColumnEncoderBin) other;
			assert other._colID == _colID;
			// save the min, max as well as the number of bins for the column indexes
			MutableTriple<Integer, Double, Double> entry = new MutableTriple<>(_numBin, _binMins[0], _binMaxs[_binMaxs.length - 1]);
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
		//allocate frame if necessary
		meta.ensureAllocatedColumns(_binMaxs.length);

		//serialize the internal state into frame meta data
		meta.getColumnMetadata(_colID-1).setNumDistinct(_numBin);
		for( int i=0; i<_binMaxs.length; i++ ) {
			StringBuilder sb = new StringBuilder(16);
			sb.append(_binMins[i]);
			sb.append(Lop.DATATYPE_PREFIX);
			sb.append(_binMaxs[i]);
			meta.set(i, _colID-1, sb.toString());
		}
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		if( meta == null || _binMaxs != null )
			return;
		//deserialize the frame meta data into internal state
		int nbins = (int)meta.getColumnMetadata()[_colID-1].getNumDistinct();
		_binMins = new double[nbins];
		_binMaxs = new double[nbins];
		for( int i=0; i<nbins; i++ ) {
			String[] tmp = meta.get(i, _colID-1).toString().split(Lop.DATATYPE_PREFIX);
			_binMins[i] = Double.parseDouble(tmp[0]);
			_binMaxs[i] = Double.parseDouble(tmp[1]);
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		
		out.writeInt(_numBin);
		out.writeBoolean(_binMaxs!=null);
		if( _binMaxs != null ){
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
}
