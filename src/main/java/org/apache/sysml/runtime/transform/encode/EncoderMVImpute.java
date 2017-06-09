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
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysml.runtime.functionobjects.CM;
import org.apache.sysml.runtime.functionobjects.Mean;
import org.apache.sysml.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

public class EncoderMVImpute extends Encoder 
{	
	private static final long serialVersionUID = 9057868620144662194L;

	public enum MVMethod { INVALID, GLOBAL_MEAN, GLOBAL_MODE, CONSTANT };
	
	private MVMethod[] _mvMethodList = null;
	private MVMethod[] _mvscMethodList = null;	// scaling methods for attributes that are imputed and also scaled
	
	private BitSet _isMVScaled = null;
	private CM _varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);		// function object that understands variance computation
	
	// objects required to compute mean and variance of all non-missing entries 
	private Mean _meanFn = Mean.getMeanFnObject();	// function object that understands mean computation
	private KahanObject[] _meanList = null; 		// column-level means, computed so far
	private long[] _countList = null;				// #of non-missing values
	
	private CM_COV_Object[] _varList = null;		// column-level variances, computed so far (for scaling)

	private int[] 			_scnomvList = null;			// List of attributes that are scaled but not imputed
	private MVMethod[]		_scnomvMethodList = null;	// scaling methods: 0 for invalid; 1 for mean-subtraction; 2 for z-scoring
	private KahanObject[] 	_scnomvMeanList = null;		// column-level means, for attributes scaled but not imputed
	private long[] 			_scnomvCountList = null;	// #of non-missing values, for attributes scaled but not imputed
	private CM_COV_Object[] _scnomvVarList = null;		// column-level variances, computed so far
	
	private String[] _replacementList = null;		// replacements: for global_mean, mean; and for global_mode, recode id of mode category
	private String[] _NAstrings = null;
	private List<Integer> _rcList = null; 
	private HashMap<Integer,HashMap<String,Long>> _hist = null;
	
	public String[] getReplacements() { return _replacementList; }
	public KahanObject[] getMeans()   { return _meanList; }
	public CM_COV_Object[] getVars()  { return _varList; }
	public KahanObject[] getMeans_scnomv()   { return _scnomvMeanList; }
	public CM_COV_Object[] getVars_scnomv()  { return _scnomvVarList; }
	
	public EncoderMVImpute(JSONObject parsedSpec, String[] colnames, int clen) 
		throws JSONException
	{
		super(null, clen);
		
		//handle column list
		int[] collist = TfMetaUtils.parseJsonObjectIDList(parsedSpec, colnames, TfUtils.TXMETHOD_IMPUTE);
		initColList(collist);
	
		//handle method list
		parseMethodsAndReplacments(parsedSpec);
		
		//create reuse histograms
		_hist = new HashMap<Integer, HashMap<String,Long>>();
	}
			
	public EncoderMVImpute(JSONObject parsedSpec, String[] colnames, String[] NAstrings, int clen)
		throws JSONException 
	{
		super(null, clen);	
		boolean isMV = parsedSpec.containsKey(TfUtils.TXMETHOD_IMPUTE);
		boolean isSC = parsedSpec.containsKey(TfUtils.TXMETHOD_SCALE);		
		_NAstrings = NAstrings;
		
		if(!isMV) {
			// MV Impute is not applicable
			_colList = null;
			_mvMethodList = null;
			_meanList = null;
			_countList = null;
			_replacementList = null;
		}
		else {
			JSONObject mvobj = (JSONObject) parsedSpec.get(TfUtils.TXMETHOD_IMPUTE);
			JSONArray mvattrs = (JSONArray) mvobj.get(TfUtils.JSON_ATTRS);
			JSONArray mvmthds = (JSONArray) mvobj.get(TfUtils.JSON_MTHD);
			int mvLength = mvattrs.size();
			
			_colList = new int[mvLength];
			_mvMethodList = new MVMethod[mvLength];
			
			_meanList = new KahanObject[mvLength];
			_countList = new long[mvLength];
			_varList = new CM_COV_Object[mvLength];
			
			_isMVScaled = new BitSet(_colList.length);
			_isMVScaled.clear();
			
			for(int i=0; i < _colList.length; i++) {
				_colList[i] = UtilFunctions.toInt(mvattrs.get(i));
				_mvMethodList[i] = MVMethod.values()[UtilFunctions.toInt(mvmthds.get(i))]; 
				_meanList[i] = new KahanObject(0, 0);
			}
			
			_replacementList = new String[mvLength]; 	// contains replacements for all columns (scale and categorical)
			
			JSONArray constants = (JSONArray)mvobj.get(TfUtils.JSON_CONSTS);
			for(int i=0; i < constants.size(); i++) {
				if ( constants.get(i) == null )
					_replacementList[i] = "NaN";
				else
					_replacementList[i] = constants.get(i).toString();
			}
		}
		
		// Handle scaled attributes
		if ( !isSC )
		{
			// scaling is not applicable
			_scnomvCountList = null;
			_scnomvMeanList = null;
			_scnomvVarList = null;
		}
		else
		{
			if ( _colList != null ) 
				_mvscMethodList = new MVMethod[_colList.length];
			
			JSONObject scobj = (JSONObject) parsedSpec.get(TfUtils.TXMETHOD_SCALE);
			JSONArray scattrs = (JSONArray) scobj.get(TfUtils.JSON_ATTRS);
			JSONArray scmthds = (JSONArray) scobj.get(TfUtils.JSON_MTHD);
			int scLength = scattrs.size();
			
			int[] _allscaled = new int[scLength];
			int scnomv = 0, colID;
			byte mthd;
			for(int i=0; i < scLength; i++)
			{
				colID = UtilFunctions.toInt(scattrs.get(i));
				mthd = (byte) UtilFunctions.toInt(scmthds.get(i)); 
						
				_allscaled[i] = colID;
				
				// check if the attribute is also MV imputed
				int mvidx = isApplicable(colID);
				if(mvidx != -1)
				{
					_isMVScaled.set(mvidx);
					_mvscMethodList[mvidx] = MVMethod.values()[mthd];
					_varList[mvidx] = new CM_COV_Object();
				}
				else
					scnomv++;	// count of scaled but not imputed 
			}
			
			if(scnomv > 0)
			{
				_scnomvList = new int[scnomv];			
				_scnomvMethodList = new MVMethod[scnomv];	
	
				_scnomvMeanList = new KahanObject[scnomv];
				_scnomvCountList = new long[scnomv];
				_scnomvVarList = new CM_COV_Object[scnomv];
				
				for(int i=0, idx=0; i < scLength; i++)
				{
					colID = UtilFunctions.toInt(scattrs.get(i));
					mthd = (byte)UtilFunctions.toInt(scmthds.get(i)); 
							
					if(isApplicable(colID) == -1)
					{	// scaled but not imputed
						_scnomvList[idx] = colID;
						_scnomvMethodList[idx] = MVMethod.values()[mthd];
						_scnomvMeanList[idx] = new KahanObject(0, 0);
						_scnomvVarList[idx] = new CM_COV_Object();
						idx++;
					}
				}
			}
		}
	}

	private void parseMethodsAndReplacments(JSONObject parsedSpec) throws JSONException {
		JSONArray mvspec = (JSONArray) parsedSpec.get(TfUtils.TXMETHOD_IMPUTE);
		_mvMethodList = new MVMethod[mvspec.size()];
		_replacementList = new String[mvspec.size()];
		_meanList = new KahanObject[mvspec.size()];
		_countList = new long[mvspec.size()];
		for(int i=0; i < mvspec.size(); i++) {
			JSONObject mvobj = (JSONObject)mvspec.get(i);
			_mvMethodList[i] = MVMethod.valueOf(mvobj.get("method").toString().toUpperCase()); 
			if( _mvMethodList[i] == MVMethod.CONSTANT ) {
				_replacementList[i] = mvobj.getString("value").toString();
			}
			_meanList[i] = new KahanObject(0, 0);
		}
	}
		
	public void prepare(String[] words) throws IOException {
		
		try {
			String w = null;
			if(_colList != null)
			for(int i=0; i <_colList.length; i++) {
				int colID = _colList[i];
				w = UtilFunctions.unquote(words[colID-1].trim());
				
				try {
				if(!TfUtils.isNA(_NAstrings, w)) {
					_countList[i]++;
					
					boolean computeMean = (_mvMethodList[i] == MVMethod.GLOBAL_MEAN || _isMVScaled.get(i) );
					if(computeMean) {
						// global_mean
						double d = UtilFunctions.parseToDouble(w);
						_meanFn.execute2(_meanList[i], d, _countList[i]);
						
						if (_isMVScaled.get(i) && _mvscMethodList[i] == MVMethod.GLOBAL_MODE)
							_varFn.execute(_varList[i], d);
					}
					else {
						// global_mode or constant
						// Nothing to do here. Mode is computed using recode maps.
					}
				}
				} catch (NumberFormatException e) 
				{
					throw new RuntimeException("Encountered \"" + w + "\" in column ID \"" + colID + "\", when expecting a numeric value. Consider adding \"" + w + "\" to na.strings, along with an appropriate imputation method.");
				}
			}
			
			// Compute mean and variance for attributes that are scaled but not imputed
			if(_scnomvList != null)
			for(int i=0; i < _scnomvList.length; i++) 
			{
				int colID = _scnomvList[i];
				w = UtilFunctions.unquote(words[colID-1].trim());
				double d = UtilFunctions.parseToDouble(w);
				_scnomvCountList[i]++; 		// not required, this is always equal to total #records processed
				_meanFn.execute2(_scnomvMeanList[i], d, _scnomvCountList[i]);
				if(_scnomvMethodList[i] == MVMethod.GLOBAL_MODE)
					_varFn.execute(_scnomvVarList[i], d);
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	public MVMethod getMethod(int colID) {
		int idx = isApplicable(colID);		
		if(idx == -1)
			return MVMethod.INVALID;
		else
			return _mvMethodList[idx];
	}
	
	public long getNonMVCount(int colID) {
		int idx = isApplicable(colID);
		return (idx == -1) ? 0 : _countList[idx];
	}
	
	public String getReplacement(int colID)  {
		int idx = isApplicable(colID);		
		return (idx == -1) ? null : _replacementList[idx];
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		build(in);
		return apply(in, out);
	}
	
	@Override
	public void build(FrameBlock in) {
		try {
			for( int j=0; j<_colList.length; j++ ) {
				int colID = _colList[j];
				if( _mvMethodList[j] == MVMethod.GLOBAL_MEAN ) {
					//compute global column mean (scale)
					long off = _countList[j];
					for( int i=0; i<in.getNumRows(); i++ )
						_meanFn.execute2(_meanList[j], UtilFunctions.objectToDouble(
							in.getSchema()[colID-1], in.get(i, colID-1)), off+i+1);
					_replacementList[j] = String.valueOf(_meanList[j]._sum);
					_countList[j] += in.getNumRows();
				}
				else if( _mvMethodList[j] == MVMethod.GLOBAL_MODE ) {
					//compute global column mode (categorical), i.e., most frequent category
					HashMap<String,Long> hist = _hist.containsKey(colID) ? 
							_hist.get(colID) : new HashMap<String,Long>();
					for( int i=0; i<in.getNumRows(); i++ ) {
						String key = String.valueOf(in.get(i, colID-1));
						if( key != null && !key.isEmpty() ) {
							Long val = hist.get(key);
							hist.put(key, (val!=null) ? val+1 : 1);
						}	
					}
					_hist.put(colID, hist);
					long max = Long.MIN_VALUE; 
					for( Entry<String, Long> e : hist.entrySet() ) 
						if( e.getValue() > max  ) {
							_replacementList[j] = e.getKey();
							max = e.getValue();
						}
				}
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	@Override
	public String[] apply(String[] words) 
	{	
		if( isApplicable() )
			for(int i=0; i < _colList.length; i++) {
				int colID = _colList[i];
				String w = UtilFunctions.unquote(words[colID-1]);
				if(TfUtils.isNA(_NAstrings, w))
					w = words[colID-1] = _replacementList[i];
				
				if ( _isMVScaled.get(i) )
					if ( _mvscMethodList[i] == MVMethod.GLOBAL_MEAN )
						words[colID-1] = Double.toString( UtilFunctions.parseToDouble(w) - _meanList[i]._sum );
					else
						words[colID-1] = Double.toString( (UtilFunctions.parseToDouble(w) - _meanList[i]._sum) / _varList[i].mean._sum );
			}
		
		if(_scnomvList != null)
		for(int i=0; i < _scnomvList.length; i++)
		{
			int colID = _scnomvList[i];
			if ( _scnomvMethodList[i] == MVMethod.GLOBAL_MEAN )
				words[colID-1] = Double.toString( UtilFunctions.parseToDouble(words[colID-1]) - _scnomvMeanList[i]._sum );
			else
				words[colID-1] = Double.toString( (UtilFunctions.parseToDouble(words[colID-1]) - _scnomvMeanList[i]._sum) / _scnomvVarList[i].mean._sum );
		}
			
		return words;
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		for(int i=0; i<in.getNumRows(); i++) {
			for(int j=0; j<_colList.length; j++) {
				int colID = _colList[j];
				if( Double.isNaN(out.quickGetValue(i, colID-1)) )
					out.quickSetValue(i, colID-1, Double.parseDouble(_replacementList[j]));
			}
		}
		return out;
	}
	
	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		for( int j=0; j<_colList.length; j++ ) {
			out.getColumnMetadata(_colList[j]-1)
			   .setMvValue(_replacementList[j]);
		}
		return out;
	}

	public void initMetaData(FrameBlock meta) {
		//init replacement lists, replace recoded values to
		//apply mv imputation potentially after recoding
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j];	
			String mvVal = UtilFunctions.unquote(meta.getColumnMetadata(colID-1).getMvValue()); 
			if( _rcList.contains(colID) ) {
				Long mvVal2 = meta.getRecodeMap(colID-1).get(mvVal);
				if( mvVal2 == null)
					throw new RuntimeException("Missing recode value for impute value '"+mvVal+"' (colID="+colID+").");
				_replacementList[j] = mvVal2.toString();
			}
			else {
				_replacementList[j] = mvVal;
			}
		}
	}

	public void initRecodeIDList(List<Integer> rcList) {
		_rcList = rcList;
	}
	
	/**
	 * Exposes the internal histogram after build.
	 * 
	 * @param colID column ID
	 * @return histogram (map of string keys and long values)
	 */
	public HashMap<String,Long> getHistogram( int colID ) {
		return _hist.get(colID);
	}
}
