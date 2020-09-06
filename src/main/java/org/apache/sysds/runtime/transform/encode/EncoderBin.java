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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderBin extends Encoder 
{
	private static final long serialVersionUID = 1917445005206076078L;

	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";

	protected int[] _numBins = null;
	
	//frame transform-apply attributes
	//TODO binMins is redundant and could be removed
	private double[][] _binMins = null;
	private double[][] _binMaxs = null;

	public EncoderBin(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException, IOException 
	{
		super( null, clen );
		if ( !parsedSpec.containsKey(TfMethod.BIN.toString()) )
			return;
		
		//parse column names or column ids
		List<Integer> collist = TfMetaUtils.parseBinningColIDs(parsedSpec, colnames, minCol, maxCol);
		initColList(ArrayUtils.toPrimitive(collist.toArray(new Integer[0])));
		
		//parse number of bins per column
		boolean ids = parsedSpec.containsKey("ids") && parsedSpec.getBoolean("ids");
		JSONArray group = (JSONArray) parsedSpec.get(TfMethod.BIN.toString());
		_numBins = new int[collist.size()];
		for (Object o : group) {
			JSONObject colspec = (JSONObject) o;
			int ixOffset = minCol == -1 ? 0 : minCol - 1;
			int pos = collist.indexOf(ids ? colspec.getInt("id") - ixOffset :
				ArrayUtils.indexOf(colnames, colspec.get("name")) + 1);
			if(pos >= 0)
				_numBins[pos] = colspec.containsKey("numbins") ? colspec.getInt("numbins") : 1;
		}
	}
	
	public EncoderBin() {
		super(new int[0], 0);
		_numBins = new int[0];
	}
	
	private EncoderBin(int[] colList, int clen, int[] numBins, double[][] binMins, double[][] binMaxs) {
		super(colList, clen);
		_numBins = numBins;
		_binMins = binMins;
		_binMaxs = binMaxs;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		build(in);
		return apply(in, out);
	}

	@Override
	public void build(FrameBlock in) {
		if ( !isApplicable() )
			return;
		// initialize internal transformation metadata
		_binMins = new double[_colList.length][];
		_binMaxs = new double[_colList.length][];
		
		// derive bin boundaries from min/max per column
		for(int j=0; j <_colList.length; j++) {
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			int colID = _colList[j];
			for( int i=0; i<in.getNumRows(); i++ ) {
				double inVal = UtilFunctions.objectToDouble(
					in.getSchema()[colID-1], in.get(i, colID-1));
				min = Math.min(min, inVal);
				max = Math.max(max, inVal);
			}
			_binMins[j] = new double[_numBins[j]];
			_binMaxs[j] = new double[_numBins[j]];
			for(int i=0; i<_numBins[j]; i++) {
				_binMins[j][i] = min + i*(max-min)/_numBins[j];
				_binMaxs[j][i] = min + (i+1)*(max-min)/_numBins[j];
			}
		}
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		for(int j=0; j<_colList.length; j++) {
			int colID = _colList[j];
			for( int i=0; i<in.getNumRows(); i++ ) {
				double inVal = UtilFunctions.objectToDouble(
						in.getSchema()[colID-1], in.get(i, colID-1));
				int ix = Arrays.binarySearch(_binMaxs[j], inVal);
				int binID = ((ix < 0) ? Math.abs(ix+1) : ix) + 1;
				out.quickSetValue(i, colID-1, binID);
			}
		}
		return out;
	}
	
	@Override
	public Encoder subRangeEncoder(IndexRange ixRange) {
		List<Integer> colsList = new ArrayList<>();
		List<Integer> numBinsList = new ArrayList<>();
		List<double[]> binMinsList = new ArrayList<>();
		List<double[]> binMaxsList = new ArrayList<>();
		for(int i = 0; i < _colList.length; i++) {
			int col = _colList[i];
			if(col >= ixRange.colStart && col < ixRange.colEnd) {
				// add the correct column, removed columns before start
				// colStart - 1 because colStart is 1-based
				int corrColumn = (int) (col - (ixRange.colStart - 1));
				colsList.add(corrColumn);
				numBinsList.add(_numBins[i]);
				binMinsList.add(_binMins[i]);
				binMaxsList.add(_binMaxs[i]);
			}
		}
		if(colsList.isEmpty())
			// empty encoder -> sub range encoder does not exist
			return null;

		int[] colList = colsList.stream().mapToInt(i -> i).toArray();
		return new EncoderBin(colList, (int) (ixRange.colEnd - ixRange.colStart),
			numBinsList.stream().mapToInt((i) -> i).toArray(), binMinsList.toArray(new double[0][0]),
			binMaxsList.toArray(new double[0][0]));
	}
	
	@Override
	public void mergeAt(Encoder other, int row, int col) {
		if(other instanceof EncoderBin) {
			EncoderBin otherBin = (EncoderBin) other;

			// save the min, max as well as the number of bins for the column indexes
			Map<Integer, MutableTriple<Integer, Double, Double>> ixBinsMap = new HashMap<>();
			for(int i = 0; i < _colList.length; i++) {
				ixBinsMap.put(_colList[i],
					new MutableTriple<>(_numBins[i], _binMins[i][0], _binMaxs[i][_binMaxs[i].length - 1]));
			}
			for(int i = 0; i < otherBin._colList.length; i++) {
				int column = otherBin._colList[i] + (col - 1);
				MutableTriple<Integer, Double, Double> entry = ixBinsMap.get(column);
				if(entry == null) {
					ixBinsMap.put(column,
						new MutableTriple<>(otherBin._numBins[i], otherBin._binMins[i][0],
							otherBin._binMaxs[i][otherBin._binMaxs[i].length - 1]));
				}
				else {
					// num bins will match
					entry.middle = Math.min(entry.middle, otherBin._binMins[i][0]);
					entry.right = Math.max(entry.right, otherBin._binMaxs[i][otherBin._binMaxs[i].length - 1]);
				}
			}

			mergeColumnInfo(other, col);

			// use the saved values to fill the arrays again
			_numBins = new int[_colList.length];
			_binMins = new double[_colList.length][];
			_binMaxs = new double[_colList.length][];

			for(int i = 0; i < _colList.length; i++) {
				int column = _colList[i];
				MutableTriple<Integer, Double, Double> entry = ixBinsMap.get(column);
				_numBins[i] = entry.left;

				double min = entry.middle;
				double max = entry.right;
				_binMins[i] = new double[_numBins[i]];
				_binMaxs[i] = new double[_numBins[i]];
				for(int j = 0; j < _numBins[i]; j++) {
					_binMins[i][j] = min + j * (max - min) / _numBins[i];
					_binMaxs[i][j] = min + (j + 1) * (max - min) / _numBins[i];
				}
			}
			return;
		}
		super.mergeAt(other, row, col);
	}
	
	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		//allocate frame if necessary
		int maxLength = 0;
		for( int j=0; j<_colList.length; j++ )
			maxLength = Math.max(maxLength, _binMaxs[j].length);
		meta.ensureAllocatedColumns(maxLength);
		
		//serialize the internal state into frame meta data
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			meta.getColumnMetadata(colID-1).setNumDistinct(_numBins[j]);
			for( int i=0; i<_binMaxs[j].length; i++ ) {
				StringBuilder sb = new StringBuilder(16);
				sb.append(_binMins[j][i]);
				sb.append(Lop.DATATYPE_PREFIX);
				sb.append(_binMaxs[j][i]);
				meta.set(i, colID-1, sb.toString());
			}
		}
		return meta;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		if( meta == null || _binMaxs != null )
			return;
		//deserialize the frame meta data into internal state
		_binMins = new double[_colList.length][];
		_binMaxs = new double[_colList.length][];
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			int nbins = (int)meta.getColumnMetadata()[colID-1].getNumDistinct();
			_binMins[j] = new double[nbins];
			_binMaxs[j] = new double[nbins];
			for( int i=0; i<nbins; i++ ) {
				String[] tmp = meta.get(i, colID-1).toString().split(Lop.DATATYPE_PREFIX);
				_binMins[j][i] = Double.parseDouble(tmp[0]);
				_binMaxs[j][i] = Double.parseDouble(tmp[1]);
			}
		}
	}
}
