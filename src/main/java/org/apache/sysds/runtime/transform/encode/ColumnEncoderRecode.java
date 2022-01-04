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
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.Callable;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.Statistics;

public class ColumnEncoderRecode extends ColumnEncoder {
	private static final long serialVersionUID = 8213163881283341874L;

	// test property to ensure consistent encoding for local and federated
	public static boolean SORT_RECODE_MAP = false;

	// recode maps and custom map for partial recode maps
	private HashMap<String, Long> _rcdMap = new HashMap<>();
	private HashSet<Object> _rcdMapPart = null;

	public ColumnEncoderRecode(int colID) {
		super(colID);
	}

	public ColumnEncoderRecode() {
		this(-1);
	}

	private ColumnEncoderRecode(int colID, HashMap<String, Long> rcdMap) {
		super(colID);
		_rcdMap = rcdMap;
	}

	/**
	 * Returns the Recode map entry which consists of concatenation of code, delimiter and token.
	 *
	 * @param token is part of Recode map
	 * @param code  is code for token
	 * @return the concatenation of token and code with delimiter in between
	 */
	public static String constructRecodeMapEntry(String token, Long code) {
		StringBuilder sb = new StringBuilder(token.length() + 16);
		return constructRecodeMapEntry(token, code, sb);
	}

	private static String constructRecodeMapEntry(String token, Long code, StringBuilder sb) {
		sb.setLength(0); // reset reused string builder
		return sb.append(token).append(Lop.DATATYPE_PREFIX).append(code.longValue()).toString();
	}

	/**
	 * Splits a Recode map entry into its token and code.
	 *
	 * @param value concatenation of token and code with delimiter in between
	 * @return string array of token and code
	 */
	public static String[] splitRecodeMapEntry(String value) {
		// Instead of using splitCSV which is forcing string with RFC-4180 format,
		// using Lop.DATATYPE_PREFIX separator to split token and code
		int pos = value.lastIndexOf(Lop.DATATYPE_PREFIX);
		return new String[] {value.substring(0, pos), value.substring(pos + 1)};
	}

	public HashMap<String, Long> getCPRecodeMaps() {
		return _rcdMap;
	}

	public HashSet<Object> getCPRecodeMapsPartial() {
		return _rcdMapPart;
	}

	public void sortCPRecodeMaps() {
		sortCPRecodeMaps(_rcdMap);
	}

	private static void sortCPRecodeMaps(HashMap<String, Long> map) {
		String[] keys = map.keySet().toArray(new String[0]);
		Arrays.sort(keys);
		map.clear();
		for(String key : keys)
			putCode(map, key);
	}

	private static void makeRcdMap(CacheBlock in, HashMap<String, Long> map, int colID, int startRow, int blk) {
		for(int row = startRow; row < getEndIndex(in.getNumRows(), startRow, blk); row++){
			String key = in.getString(row, colID - 1);
			if(key != null && !key.isEmpty() && !map.containsKey(key))
				putCode(map, key);
		}
		if(SORT_RECODE_MAP) {
			sortCPRecodeMaps(map);
		}
	}

	private long lookupRCDMap(String key) {
		Long tmp = _rcdMap.get(key);
		return (tmp != null) ? tmp : -1;
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.RECODE;
	}

	@Override
	public void build(CacheBlock in) {
		if(!isApplicable())
			return;
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		makeRcdMap(in, _rcdMap, _colID, 0, in.getNumRows());
		if(DMLScript.STATISTICS){
			Statistics.incTransformRecodeBuildTime(System.nanoTime() - t0);
		}
	}

	@Override
	public Callable<Object> getBuildTask(CacheBlock in) {
		return new ColumnRecodeBuildTask(this, in);
	}

	@Override
	public Callable<Object> getPartialBuildTask(CacheBlock in, int startRow, 
			int blockSize, HashMap<Integer, Object> ret) {
		return new RecodePartialBuildTask(in, _colID, startRow, blockSize, ret);
	}

	@Override
	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret) {
		return new RecodeMergePartialBuildTask(this, ret);
	}

	/**
	 * Put the code into the map with the provided key. The code depends on the type of encoder.
	 *
	 * @param map column map
	 * @param key key for the new entry
	 */
	protected static void putCode(HashMap<String, Long> map, String key) {
		map.put(key, (long) (map.size() + 1));
	}

	protected double getCode(CacheBlock in, int r){
		// lookup for a single row
		Object okey = in.getString(r, _colID - 1);
		String key = (okey != null) ? okey.toString() : null;
		if(key == null || key.isEmpty())
			return Double.NaN;
		long code = lookupRCDMap(key);
		return (code < 0) ? Double.NaN : code;
	}
	
	@Override
	protected double[] getCodeCol(CacheBlock in, int startInd, int blkSize) {
		// lookup for a block of rows
		int endInd = getEndIndex(in.getNumRows(), startInd, blkSize);
		double codes[] = new double[endInd-startInd];
		for (int i=startInd; i<endInd; i++) {
			String key = in.getString(i, _colID-1);
			if(key == null || key.isEmpty()) {
				codes[i-startInd] = Double.NaN;
				continue;
			}
			long code = lookupRCDMap(key);
			codes[i-startInd] = (code < 0) ? Double.NaN : code;
		}
		return codes;
	}

	@Override
	public void prepareBuildPartial() {
		// ensure allocated partial recode map
		if(_rcdMapPart == null)
			_rcdMapPart = new HashSet<>();
	}

	@Override
	public void buildPartial(FrameBlock in) {
		if(!isApplicable())
			return;

		// construct partial recode map (tokens w/o codes)
		// probe and build column map
		for(int i = 0; i < in.getNumRows(); i++)
			_rcdMapPart.add(in.get(i, _colID - 1));
		// cleanup unnecessary entries once
		_rcdMapPart.remove(null);
		_rcdMapPart.remove("");
	}


	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(CacheBlock in, MatrixBlock out, int outputCol, int startRow, int blk){
		return new RecodeSparseApplyTask(this, in ,out, outputCol, startRow, blk);
	}

	@Override
	public void mergeAt(ColumnEncoder other) {
		if(!(other instanceof ColumnEncoderRecode)) {
			super.mergeAt(other);
			return;
		}
		assert other._colID == _colID;
		// merge together overlapping columns
		ColumnEncoderRecode otherRec = (ColumnEncoderRecode) other;
		HashMap<String, Long> otherMap = otherRec._rcdMap;
		if(otherMap != null) {
			// for each column, add all non present recode values
			for(Map.Entry<String, Long> entry : otherMap.entrySet()) {
				if(lookupRCDMap(entry.getKey()) == -1) {
					// key does not yet exist
					putCode(_rcdMap, entry.getKey());
				}
			}
		}
	}

	public int getNumDistinctValues() {
		return _rcdMap.size();
	}
	
	@Override
	public void allocateMetaData(FrameBlock meta) {
		// allocate output rows
		meta.ensureAllocatedColumns(getNumDistinctValues());
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		if(!isApplicable())
			return meta;

		// inverse operation to initRecodeMaps

		// allocate output rows
		meta.ensureAllocatedColumns(getNumDistinctValues());

		// create compact meta data representation
		StringBuilder sb = new StringBuilder(); // for reuse
		int rowID = 0;
		for(Entry<String, Long> e : _rcdMap.entrySet()) {
			meta.set(rowID++, _colID - 1, // 1-based
				constructRecodeMapEntry(e.getKey(), e.getValue(), sb));
		}
		meta.getColumnMetadata(_colID - 1).setNumDistinct(getNumDistinctValues());

		return meta;
	}

	/**
	 * Construct the recodemaps from the given input frame for all columns registered for recode.
	 *
	 * @param meta frame block
	 */
	@Override
	public void initMetaData(FrameBlock meta) {
		if(meta == null || meta.getNumRows() <= 0)
			return;
		_rcdMap = meta.getRecodeMap(_colID - 1); // 1-based
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeInt(_rcdMap.size());
		for(Entry<String, Long> e : _rcdMap.entrySet()) {
			out.writeUTF(e.getKey());
			out.writeLong(e.getValue());
		}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		int size = in.readInt();
		for(int j = 0; j < size; j++) {
			String key = in.readUTF();
			Long value = in.readLong();
			_rcdMap.put(key, value);
		}
	}

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		ColumnEncoderRecode that = (ColumnEncoderRecode) o;
		return Objects.equals(_rcdMap, that._rcdMap);
	}

	@Override
	public int hashCode() {
		return Objects.hash(_rcdMap);
	}

	public HashMap<String, Long> getRcdMap() {
		return _rcdMap;
	}

	private static class RecodeSparseApplyTask extends ColumnApplyTask<ColumnEncoderRecode>{

		public RecodeSparseApplyTask(ColumnEncoderRecode encoder, CacheBlock input, MatrixBlock out, int outputCol) {
			super(encoder, input, out, outputCol);
		}

		protected RecodeSparseApplyTask(ColumnEncoderRecode encoder, CacheBlock input, MatrixBlock out,
										int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		public Object call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if(_out.getSparseBlock() == null)
				return null;
			_encoder.applySparse(_input, _out, _outputCol, _startRow, _blk);
			if(DMLScript.STATISTICS){
				Statistics.incTransformRecodeApplyTime(System.nanoTime() - t0);
			}
			return null;
		}

		@Override
		public String toString() {
			String str = getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
			if(_blk != -1)
				str+= "<Sr: " + _startRow + ">";
			return str;
		}

	}

	private static class RecodePartialBuildTask implements Callable<Object> {

		private final CacheBlock _input;
		private final int _blockSize;
		private final int _startRow;
		private final int _colID;
		private final HashMap<Integer, Object> _partialMaps;

		protected RecodePartialBuildTask(CacheBlock input, int colID, int startRow, 
				int blocksize, HashMap<Integer, Object> partialMaps) {
			_input = input;
			_blockSize = blocksize;
			_colID = colID;
			_startRow = startRow;
			_partialMaps = partialMaps;
		}

		@Override
		public HashMap<String, Long> call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			HashMap<String, Long> partialMap = new HashMap<>();
			makeRcdMap(_input, partialMap, _colID, _startRow, _blockSize);
			synchronized(_partialMaps) {
				_partialMaps.put(_startRow, partialMap);
			}
			if(DMLScript.STATISTICS){
				Statistics.incTransformRecodeBuildTime(System.nanoTime() - t0);
			}
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<Start row: " + _startRow + "; Block size: " + _blockSize + ">";
		}

	}

	private static class RecodeMergePartialBuildTask implements Callable<Object> {
		private final HashMap<Integer, ?> _partialMaps;
		private final ColumnEncoderRecode _encoder;

		private RecodeMergePartialBuildTask(ColumnEncoderRecode encoderRecode, HashMap<Integer, ?> partialMaps) {
			_partialMaps = partialMaps;
			_encoder = encoderRecode;
		}

		@Override
		public Object call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			HashMap<String, Long> rcdMap = _encoder.getRcdMap();
			_partialMaps.forEach((start_row, map) -> {
				((HashMap<?, ?>) map).forEach((k, v) -> {
					if(!rcdMap.containsKey((String) k))
						putCode(rcdMap, (String) k);
				});
			});
			_encoder._rcdMap = rcdMap;
			if(DMLScript.STATISTICS){
				Statistics.incTransformRecodeBuildTime(System.nanoTime() - t0);
			}
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}

	}

	private static class ColumnRecodeBuildTask implements Callable<Object> {

		private final ColumnEncoderRecode _encoder;
		private final CacheBlock _input;

		protected ColumnRecodeBuildTask(ColumnEncoderRecode encoder, CacheBlock input) {
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
