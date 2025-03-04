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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.TransformStatistics;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

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

public class EncoderMVImpute extends LegacyEncoder {
	private static final long serialVersionUID = 9057868620144662194L;
	// objects required to compute mean and variance of all non-missing entries
	private final Mean _meanFn = Mean.getMeanFnObject(); // function object that understands mean computation
	private MVMethod[] _mvMethodList = null;
	private KahanObject[] _meanList = null; // column-level means, computed so far
	private long[] _countList = null; // #of non-missing values
	private String[] _replacementList = null; // replacements: for global_mean, mean; and for global_mode, recode id of
												// mode category
	private List<Integer> _rcList = null;
	private HashMap<Integer, HashMap<String, Long>> _hist = null;
	public EncoderMVImpute(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException {
		super(null, clen);

		// handle column list
		int[] collist = TfMetaUtils
			.parseJsonObjectIDList(parsedSpec, colnames, TfMethod.IMPUTE.toString(), minCol, maxCol);
		initColList(collist);

		// handle method list
		parseMethodsAndReplacements(parsedSpec, colnames, minCol);

		// create reuse histograms
		_hist = new HashMap<>();
	}

	public EncoderMVImpute() {
		super(new int[0], 0);
	}

	public EncoderMVImpute(int[] colList, MVMethod[] mvMethodList, String[] replacementList, KahanObject[] meanList,
		long[] countList, List<Integer> rcList, int clen) {
		super(colList, clen);
		_mvMethodList = mvMethodList;
		_replacementList = replacementList;
		_meanList = meanList;
		_countList = countList;
		_rcList = rcList;
	}

	private static void fillListsFromMap(Map<Integer, ColInfo> map, int[] colList, MVMethod[] mvMethodList,
		String[] replacementList, KahanObject[] meanList, long[] countList,
		HashMap<Integer, HashMap<String, Long>> hist) {
		int i = 0;
		for(Entry<Integer, ColInfo> entry : map.entrySet()) {
			colList[i] = entry.getKey();
			mvMethodList[i] = entry.getValue()._method;
			replacementList[i] = entry.getValue()._replacement;
			meanList[i] = entry.getValue()._mean;
			countList[i++] = entry.getValue()._count;

			hist.put(entry.getKey(), entry.getValue()._hist);
		}
	}

	public String[] getReplacements() {
		return _replacementList;
	}

	public KahanObject[] getMeans() {
		return _meanList;
	}

	private void parseMethodsAndReplacements(JSONObject parsedSpec, String[] colnames, int offset)
		throws JSONException {
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

	public String getReplacement(int colID) {
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
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		try {
			for(int j = 0; j < _colList.length; j++) {
				int colID = _colList[j];
				if(_mvMethodList[j] == MVMethod.GLOBAL_MEAN) {
					// compute global column mean (scale)
					long off = _countList[j];
					for(int i = 0; i < in.getNumRows(); i++) {
						Object key = in.get(i, colID - 1);
						if(key == null) {
							off--;
							continue;
						}
						_meanFn.execute2(_meanList[j],
							UtilFunctions.objectToDouble(in.getSchema()[colID - 1], key),
							off + i + 1);
					}
					_replacementList[j] = String.valueOf(_meanList[j]._sum);
					_countList[j] += in.getNumRows();
				}
				else if(_mvMethodList[j] == MVMethod.GLOBAL_MODE) {
					// compute global column mode (categorical), i.e., most frequent category
					HashMap<String, Long> hist = _hist.containsKey(colID) ? _hist.get(colID) : new HashMap<>();
					for(int i = 0; i < in.getNumRows(); i++) {
						String key = String.valueOf(in.get(i, colID - 1));
						if(!key.equals("null") && !key.isEmpty()) {
							Long val = hist.get(key);
							hist.put(key, (val != null) ? val + 1 : 1);
						}
					}
					_hist.put(colID, hist);
					long max = Long.MIN_VALUE;
					for(Entry<String, Long> e : hist.entrySet())
						if(e.getValue() > max) {
							_replacementList[j] = e.getKey();
							max = e.getValue();
						}
				}
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		if(DMLScript.STATISTICS)
			TransformStatistics.incImputeBuildTime(System.nanoTime()-t0);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		for(int i = 0; i < in.getNumRows(); i++) {
			for(int j = 0; j < _colList.length; j++) {
				int colID = _colList[j];
				if(Double.isNaN(out.get(i, colID - 1)))
					out.set(i, colID - 1, Double.parseDouble(_replacementList[j]));
			}
		}
		if(DMLScript.STATISTICS)
			TransformStatistics.incImputeApplyTime(System.nanoTime()-t0);
		return out;
	}

	@Override
	public LegacyEncoder subRangeEncoder(IndexRange ixRange) {
		Map<Integer, ColInfo> map = new HashMap<>();
		for(int i = 0; i < _colList.length; i++) {
			int col = _colList[i];
			if(ixRange.inColRange(col))
				map.put(_colList[i],
					new ColInfo(_mvMethodList[i], _replacementList[i], _meanList[i], _countList[i], _hist.get(i)));
		}
		if(map.size() == 0)
			// empty encoder -> sub range encoder does not exist
			return null;

		int[] colList = new int[map.size()];
		MVMethod[] mvMethodList = new MVMethod[map.size()];
		String[] replacementList = new String[map.size()];
		KahanObject[] meanList = new KahanObject[map.size()];
		long[] countList = new long[map.size()];

		fillListsFromMap(map, colList, mvMethodList, replacementList, meanList, countList, _hist);

		if(_rcList == null)
			_rcList = new ArrayList<>();
		 
		List<Integer> rcList = _rcList.stream() //
			.filter((x) -> ixRange.inColRange(x)) //
			.map(i -> (int) (i - (ixRange.colStart - 1))) //
			.collect(Collectors.toList());

		return new EncoderMVImpute(colList, mvMethodList, replacementList, meanList, countList, rcList,
			(int) ixRange.colSpan());
	}

	@Override
	public void mergeAt(LegacyEncoder other, int row, int col) {
		if(other instanceof EncoderMVImpute) {
			EncoderMVImpute otherImpute = (EncoderMVImpute) other;
			Map<Integer, ColInfo> map = new HashMap<>();
			for(int i = 0; i < _colList.length; i++) {
				map.put(_colList[i],
					new ColInfo(_mvMethodList[i], _replacementList[i], _meanList[i], _countList[i], _hist.get(i + 1)));
			}
			for(int i = 0; i < other._colList.length; i++) {
				int column = other._colList[i];
				ColInfo otherColInfo = new ColInfo(otherImpute._mvMethodList[i], otherImpute._replacementList[i],
					otherImpute._meanList[i], otherImpute._countList[i], otherImpute._hist.get(i + 1));
				ColInfo colInfo = map.get(column);
				if(colInfo == null)
					map.put(column, otherColInfo);
				else
					colInfo.merge(otherColInfo);
			}

			_colList = new int[map.size()];
			_mvMethodList = new MVMethod[map.size()];
			_replacementList = new String[map.size()];
			_meanList = new KahanObject[map.size()];
			_countList = new long[map.size()];
			_hist = new HashMap<>();

			fillListsFromMap(map, _colList, _mvMethodList, _replacementList, _meanList, _countList, _hist);

			if(_rcList == null)
				_rcList = new ArrayList<>();
			Set<Integer> rcSet = new HashSet<>(_rcList);
			rcSet.addAll(otherImpute._rcList.stream().map(i -> i + (col - 1)).collect(Collectors.toSet()));
			_rcList = new ArrayList<>(rcSet);
			return;
		}
		super.mergeAt(other, row, col);
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		for(int j = 0; j < _colList.length; j++) {
			out.getColumnMetadata(_colList[j] - 1).setMvValue(_replacementList[j]);
		}
		return out;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		// init replacement lists, replace recoded values to
		// apply mv imputation potentially after recoding
		for(int j = 0; j < _colList.length; j++) {
			int colID = _colList[j];
			String mvVal = UtilFunctions.unquote(meta.getColumnMetadata(colID - 1).getMvValue());
			if(_rcList.contains(colID)) {
				Integer mvVal2 = meta.getRecodeMap(colID - 1).get(mvVal);
				if(mvVal2 == null)
					throw new RuntimeException(
						"Missing recode value for impute value '" + mvVal + "' (colID=" + colID + ").");
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
	public HashMap<String, Long> getHistogram(int colID) {
		return _hist.get(colID);
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		for(int i = 0; i < _colList.length; i++) {
			out.writeByte(_mvMethodList[i].ordinal());
			out.writeLong(_countList[i]);
		}

		List<String> notNullReplacements = new ArrayList<>(Arrays.asList(_replacementList));
		notNullReplacements.removeAll(Collections.singleton(null));
		out.writeInt(notNullReplacements.size());
		for(int i = 0; i < _replacementList.length; i++)
			if(_replacementList[i] != null) {
				out.writeInt(i);
				out.writeUTF(_replacementList[i]);
			}

		out.writeInt(_rcList.size());
		for(int rc : _rcList)
			out.writeInt(rc);

		int histSize = _hist == null ? 0 : _hist.size();
		out.writeInt(histSize);
		if(histSize > 0)
			for(Entry<Integer, HashMap<String, Long>> e1 : _hist.entrySet()) {
				out.writeInt(e1.getKey());
				out.writeInt(e1.getValue().size());
				for(Entry<String, Long> e2 : e1.getValue().entrySet()) {
					out.writeUTF(e2.getKey());
					out.writeLong(e2.getValue());
				}
			}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);

		_mvMethodList = new MVMethod[_colList.length];
		_countList = new long[_colList.length];
		_meanList = new KahanObject[_colList.length];
		_replacementList = new String[_colList.length];

		for(int i = 0; i < _colList.length; i++) {
			_mvMethodList[i] = MVMethod.values()[in.readByte()];
			_countList[i] = in.readLong();
			_meanList[i] = new KahanObject(0, 0);
		}

		int size4 = in.readInt();
		for(int i = 0; i < size4; i++) {
			int index = in.readInt();
			_replacementList[index] = in.readUTF();
		}

		int size3 = in.readInt();
		_rcList = new ArrayList<>();
		for(int j = 0; j < size3; j++)
			_rcList.add(in.readInt());

		_hist = new HashMap<>();
		int size1 = in.readInt();
		for(int i = 0; i < size1; i++) {
			Integer key1 = in.readInt();
			int size2 = in.readInt();

			HashMap<String, Long> maps = new HashMap<>();
			for(int j = 0; j < size2; j++) {
				String key2 = in.readUTF();
				Long value = in.readLong();
				maps.put(key2, value);
			}
			_hist.put(key1, maps);
		}
	}

	public enum MVMethod {
		INVALID, GLOBAL_MEAN, GLOBAL_MODE, CONSTANT
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
					if(_hist == null)
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