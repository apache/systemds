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
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

public class EncoderBin extends Encoder 
{
	private static final long serialVersionUID = 1917445005206076078L;

	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";

	private int[] _numBins = null;
	
	//frame transform-apply attributes
	//TODO binMins is redundant and could be removed
	private double[][] _binMins = null;
	private double[][] _binMaxs = null;

	public EncoderBin(JSONObject parsedSpec, String[] colnames, int clen) 
		throws JSONException, IOException 
	{
		super( null, clen );
		if ( !parsedSpec.containsKey(TfMethod.BIN.toString()) )
			return;
		
		//parse column names or column ids
		List<Integer> collist = TfMetaUtils.parseBinningColIDs(parsedSpec, colnames);
		initColList(ArrayUtils.toPrimitive(collist.toArray(new Integer[0])));
		
		//parse number of bins per column
		boolean ids = parsedSpec.containsKey("ids") && parsedSpec.getBoolean("ids");
		JSONArray group = (JSONArray) parsedSpec.get(TfMethod.BIN.toString());
		_numBins = new int[collist.size()];
		for(int i=0; i < _numBins.length; i++) {
			JSONObject colspec = (JSONObject) group.get(i);
			int pos = collist.indexOf(ids ? colspec.getInt("id") :
				ArrayUtils.indexOf(colnames, colspec.get("name"))+1);
			_numBins[pos] = colspec.containsKey("numbins") ?
				colspec.getInt("numbins"): 1;
		}
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
