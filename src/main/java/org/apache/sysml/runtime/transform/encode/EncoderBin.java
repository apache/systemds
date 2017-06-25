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

package org.apache.sysml.runtime.transform.encode;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

public class EncoderBin extends Encoder 
{	
	private static final long serialVersionUID = 1917445005206076078L;

	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";

	private int[] _numBins = null;
	private double[] _min=null, _max=null;	// min and max among non-missing values
	private double[] _binWidths = null;		// width of a bin for each attribute
	
	//frame transform-apply attributes
	private double[][] _binMins = null;
	private double[][] _binMaxs = null;
	
	public EncoderBin(JSONObject parsedSpec, String[] colnames, int clen) 
		throws JSONException, IOException 
	{
		this(parsedSpec, colnames, clen, false);
	}

	public EncoderBin(JSONObject parsedSpec, String[] colnames, int clen, boolean colsOnly) 
		throws JSONException, IOException 
	{
		super( null, clen );		
		if ( !parsedSpec.containsKey(TfUtils.TXMETHOD_BIN) )
			return;
		
		if( colsOnly ) {
			List<Integer> collist = TfMetaUtils.parseBinningColIDs(parsedSpec, colnames);
			initColList(ArrayUtils.toPrimitive(collist.toArray(new Integer[0])));
		}
		else 
		{
			JSONObject obj = (JSONObject) parsedSpec.get(TfUtils.TXMETHOD_BIN);		
			JSONArray attrs = (JSONArray) obj.get(TfUtils.JSON_ATTRS);
			JSONArray nbins = (JSONArray) obj.get(TfUtils.JSON_NBINS);
			initColList(attrs);
			
			_numBins = new int[attrs.size()];
			for(int i=0; i < _numBins.length; i++)
				_numBins[i] = UtilFunctions.toInt(nbins.get(i)); 
			
			// initialize internal transformation metadata
			_min = new double[_colList.length];
			Arrays.fill(_min, Double.MAX_VALUE);
			_max = new double[_colList.length];
			Arrays.fill(_max, -Double.MAX_VALUE);
			
			_binWidths = new double[_colList.length];
		}
	}

	public void prepare(String[] words, TfUtils agents) {
		if ( !isApplicable() )
			return;
		
		for(int i=0; i <_colList.length; i++) {
			int colID = _colList[i];
			
			String w = null;
			double d = 0;
				
			// equi-width
			w = UtilFunctions.unquote(words[colID-1].trim());
			if(!TfUtils.isNA(agents.getNAStrings(),w)) {
				d = UtilFunctions.parseToDouble(w);
				if(d < _min[i])
					_min[i] = d;
				if(d > _max[i])
					_max[i] = d;
			}
		}
	}
		
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		build(in);
		return apply(in, out);
	}

	@Override
	public void build(FrameBlock in) {
		// nothing to do
	}
	
	/**
	 * Method to apply transformations.
	 */
	@Override
	public String[] apply(String[] words) {
		if( !isApplicable() )
			return words;
	
		for(int i=0; i < _colList.length; i++) {
			int colID = _colList[i];
			try {
				double val = UtilFunctions.parseToDouble(words[colID-1]);
				int binid = 1;
				double tmp = _min[i] + _binWidths[i];
				while(val > tmp && binid < _numBins[i]) {
					tmp += _binWidths[i];
					binid++;
				}
				words[colID-1] = Integer.toString(binid);
			} 
			catch(NumberFormatException e) {
				throw new RuntimeException("Encountered \"" + words[colID-1] + "\" in column ID \"" + colID + "\", when expecting a numeric value. Consider adding \"" + words[colID-1] + "\" to na.strings, along with an appropriate imputation method.");
			}
		}
		
		return words;
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
		return meta;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
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
