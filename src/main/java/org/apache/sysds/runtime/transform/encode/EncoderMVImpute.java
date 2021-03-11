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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderMVImpute extends Encoder {
	private static final long serialVersionUID = 9057868620144662194L;

	public enum MVMethod { INVALID, GLOBAL_MEAN, GLOBAL_MODE, CONSTANT }
	
	private MVMethod _mvMethod = null;
	
	// objects required to compute mean and variance of all non-missing entries 
	private final Mean _meanFn = Mean.getMeanFnObject();  // function object that understands mean computation
	private KahanObject _mean = null;         // column-level means, computed so far
	private long _count = -1;               // #of non-missing values
	
	private String _replacement = null; // replacements: for global_mean, mean; and for global_mode, recode id of mode category
	private boolean _rc = false;
	private HashMap<String,Long> _hist = null;

	public String getReplacements() { return _replacement; }
	public KahanObject getMeans()   { return _mean; }
	
	public EncoderMVImpute() {
		super(-1);
	}
	
	
	public EncoderMVImpute(int colID, MVMethod mvMethod, String replacement, KahanObject mean,
						   long count, List<Integer> rcList) {
		super(colID);
		_mvMethod = mvMethod;
		_replacement = replacement;
		_mean = mean;
		_count = count;
		initRecodeIDList(rcList);
		_hist = new HashMap<>();
	}

	/*
	private void parseMethodsAndReplacements(JSONObject parsedSpec, String[] colnames, int offset) throws JSONException {
		JSONArray mvspec = (JSONArray) parsedSpec.get(TfMethod.IMPUTE.toString());
		boolean ids = parsedSpec.containsKey("ids") && parsedSpec.getBoolean("ids");
		// make space for all elements
		_mvMethodList = new MVMethod[mvspec.size()];
		_replacementList = new String[mvspec.size()];
		_meanList = new KahanObject[mvspec.size()];
		_countList = new long[mvspec.size()];
		// sort for binary search
		Arrays.sort(_colList);
		
		int listIx = 0;
		for(Object o : mvspec) {
			JSONObject mvobj = (JSONObject) o;
			int ixOffset = offset == -1 ? 0 : offset - 1;
			// check for position -> -1 if not present
			int pos = Arrays.binarySearch(_colList,
				ids ? mvobj.getInt("id") - ixOffset : ArrayUtils.indexOf(colnames, mvobj.get("name")) + 1);
			if(pos >= 0) {
				// add to arrays
				_mvMethodList[listIx] = MVMethod.valueOf(mvobj.get("method").toString().toUpperCase());
				if(_mvMethodList[listIx] == MVMethod.CONSTANT) {
					_replacementList[listIx] = mvobj.getString("value");
				}
				_meanList[listIx++] = new KahanObject(0, 0);
			}
		}
		// make arrays required size
		_mvMethodList = Arrays.copyOf(_mvMethodList, listIx);
		_replacementList = Arrays.copyOf(_replacementList, listIx);
		_meanList = Arrays.copyOf(_meanList, listIx);
		_countList = Arrays.copyOf(_countList, listIx);
	}

	 */
	
	public MVMethod getMethod(int colID) {
		if(!isApplicable(colID))
			return MVMethod.INVALID;
		else
			return _mvMethod;
	}
	
	public long getNonMVCount(int colID) {
		return isApplicable(colID) ? _count: 0;
	}
	
	public String getReplacement(int colID) {
		return isApplicable(colID) ? _replacement : null;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		build(in);
		return apply(in, out);
	}
	
	@Override
	public void build(FrameBlock in) {
		try {
			if( _mvMethod == MVMethod.GLOBAL_MEAN ) {
				//compute global column mean (scale)
				long off = _count;
				for( int i=0; i<in.getNumRows(); i++ ){
					Object key = in.get(i, _colID-1);
					if(key == null){
						off--;
						continue;
					}
					_meanFn.execute2(_mean, UtilFunctions.objectToDouble(
							in.getSchema()[_colID-1], key), off+i+1);
				}
				_replacement = String.valueOf(_mean._sum);
				_count += in.getNumRows();
			}
			else if( _mvMethod == MVMethod.GLOBAL_MODE ) {
				//compute global column mode (categorical), i.e., most frequent category
				for( int i=0; i<in.getNumRows(); i++ ) {
					String key = String.valueOf(in.get(i, _colID-1));
					if(!key.equals("null") && !key.isEmpty() ) {
						Long val = _hist.get(key);
						_hist.put(key, (val!=null) ? val+1 : 1);
					}
				}
				long max = Long.MIN_VALUE;
				for( Entry<String, Long> e : _hist.entrySet() )
					if( e.getValue() > max  ) {
						_replacement = e.getKey();
						max = e.getValue();
					}
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		for(int i=0; i<in.getNumRows(); i++) {
			if( Double.isNaN(out.quickGetValue(i, _colID-1)) )
				out.quickSetValue(i, _colID-1, Double.parseDouble(_replacement));
		}
		return out;
	}

	@Override
	public void mergeAt(Encoder other, int row) {
		if(!(other instanceof EncoderMVImpute)) {
			super.mergeAt(other, row);
			return;
		}
		EncoderMVImpute otherImpute = (EncoderMVImpute) other;
		assert otherImpute._colID == _colID;
		assert otherImpute._mvMethod == _mvMethod;

		ColInfo colInfo = new ColInfo(_mvMethod, _replacement, _mean, _count, _hist);
		ColInfo otherColInfo = new ColInfo(otherImpute._mvMethod, otherImpute._replacement, otherImpute._mean,
				otherImpute._count, otherImpute._hist);
		colInfo.merge(otherColInfo);

		_rc = _rc || otherImpute._rc;
		_replacement = colInfo._replacement;
		_mean = colInfo._mean;
		_count = colInfo._count;
		_hist = colInfo._hist;
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		out.getColumnMetadata(_colID-1).setMvValue(_replacement);
		return out;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		//init replacement lists, replace recoded values to
		//apply mv imputation potentially after recoding
		String mvVal = UtilFunctions.unquote(meta.getColumnMetadata(_colID-1).getMvValue());
		if(_rc) {
			Long mvVal2 = meta.getRecodeMap(_colID-1).get(mvVal);
			if( mvVal2 == null)
				throw new RuntimeException("Missing recode value for impute value '"+mvVal+"' (colID="+_colID+").");
			_replacement = mvVal2.toString();
		}
		else {
			_replacement = mvVal;
		}
	}

	public void initRecodeIDList(List<Integer> rcList) {
		_rc = rcList.contains(_colID);
	}
	
	/**
	 * Exposes the internal histogram after build.
	 * 
	 * @param colID column ID
	 * @return histogram (map of string keys and long values)
	 */
	public HashMap<String,Long> getHistogram() {
		return _hist;
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeByte(_mvMethod.ordinal());
		out.writeLong(_count);

		if(_replacement != null) {
			out.writeUTF(_replacement);
		}
		out.writeBoolean(_rc);
		int histSize = _hist == null ? 0 : _hist.size();
		out.writeInt(histSize);
		if (histSize > 0){
			for(Entry<String, Long> e : _hist.entrySet()) {
				out.writeUTF(e.getKey());
				out.writeLong(e.getValue());
			}
		}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);

		_mvMethod = MVMethod.values()[in.readByte()];
		_count = in.readLong();
		_mean = new KahanObject(0, 0);

		_replacement = in.readUTF();

		_rc = in.readBoolean();

		_hist = new HashMap<>();
		int size = in.readInt();
		for(int j = 0; j < size; j++){
			String key = in.readUTF();
			Long value = in.readLong();
			_hist.put(key, value);
		}

	}
	
	private static class ColInfo {
		MVMethod _method;
		String _replacement;
		KahanObject _mean;
		long _count;
		HashMap<String, Long> _hist;

		ColInfo(MVMethod method, String replacement, KahanObject mean, long count, HashMap<String, Long> hist) {
			_method = method;
			_replacement = replacement;
			_mean = mean;
			_count = count;
			_hist = hist;
		}

		public void merge(ColInfo otherColInfo) {
			if(_method != otherColInfo._method)
				throw new DMLRuntimeException("Tried to merge two different impute methods: " + _method.name() + " vs. "
						+ otherColInfo._method.name());
			switch(_method) {
				case CONSTANT:
					assert _replacement.equals(otherColInfo._replacement);
					break;
				case GLOBAL_MEAN:
					_mean._sum *= _count;
					_mean._correction *= _count;
					KahanPlus.getKahanPlusFnObject().execute(_mean, otherColInfo._mean._sum * otherColInfo._count);
					KahanPlus.getKahanPlusFnObject().execute(_mean,
						otherColInfo._mean._correction * otherColInfo._count);
					_count += otherColInfo._count;
					break;
				case GLOBAL_MODE:
					if (_hist == null)
						_hist = new HashMap<>(otherColInfo._hist);
					else
						// add counts
						_hist.replaceAll((key, count) -> count + otherColInfo._hist.getOrDefault(key, 0L));
					break;
				default:
					throw new DMLRuntimeException("Method `" + _method.name() + "` not supported for federated impute");
			}
		}
	}
}
