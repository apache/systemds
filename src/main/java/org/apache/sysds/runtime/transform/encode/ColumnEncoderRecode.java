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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

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

	private static void makeRcdMap(FrameBlock in, HashMap<String, Long> map, int colID, int startRow, int blk) {
		Iterator<String[]> iter = in.getStringRowIterator(startRow, getEndIndex(in.getNumRows(), startRow, blk), colID);
		while(iter.hasNext()) {
			String[] row = iter.next();
			// probe and build column map
			String key = row[0]; // 0 since there is only one column in the row
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
	public void build(FrameBlock in) {
		if(!isApplicable())
			return;
		makeRcdMap(in, _rcdMap, _colID, 0, in.getNumRows());
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(FrameBlock in, int blockSize){
		List<Callable<Object>> tasks = new ArrayList<>();
		List<List<? extends Callable<?>>> dep = null;
		if(blockSize == -1 || blockSize >= in.getNumRows()){
			tasks.add(new ColumnRecodeBuildTask(this, in));
		}else{
			HashMap<Integer, HashMap<String, Long>> partialMaps = new HashMap<>();
			for(int i = 0; i < in.getNumRows(); i = i + blockSize)
				tasks.add(new RecodePartialBuildTask(in, _colID, i, blockSize, partialMaps));
			if(in.getNumRows() % blockSize != 0)
				tasks.add(new RecodePartialBuildTask(in, _colID, in.getNumRows() - in.getNumRows() % blockSize,
						-1, partialMaps));
			tasks.add(new RecodeMergePartialBuildTask(this, partialMaps));
			dep = new ArrayList<>(Collections.nCopies(tasks.size()-1, null));
			dep.add(tasks.subList(0, tasks.size()-1));
		}
		return DependencyThreadPool.createDependencyTasks(tasks, dep);
	}
	
	
	@Override
	public List<Callable<Object>> getPartialBuildTasks(FrameBlock in, int blockSize) {
		List<Callable<Object>> tasks = new ArrayList<>();
		for(int i = 0; i < in.getNumRows(); i = i + blockSize)
			tasks.add(new RecodePartialBuildTask(in, _colID, i, blockSize));
		if(in.getNumRows() % blockSize != 0)
			tasks.add(new RecodePartialBuildTask(in, _colID, in.getNumRows() - in.getNumRows() % blockSize, -1));
		return tasks;
	}

	@Override
	public void mergeBuildPartial(List<Future<Object>> futurePartials, int start, int end)
		throws ExecutionException, InterruptedException {
		for(int i = start; i < end; i++) {
			Object partial = futurePartials.get(i).get();
			if(!(partial instanceof HashMap)) {
				throw new DMLRuntimeException(
					"Tried to merge " + partial.getClass() + " object into RecodeEncoder. " + "HashMap was expected.");
			}
			HashMap<?, ?> partialMap = (HashMap<?, ?>) partial;
			partialMap.forEach((k, v) -> {
				if(!_rcdMap.containsKey((String) k))
					putCode(_rcdMap, (String) k);
			});
		}
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
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		// FrameBlock is column Major and MatrixBlock row Major this results in cache inefficiencies :(
		for(int i = rowStart; i < getEndIndex(in.getNumRows(), rowStart, blk); i++) {
			Object okey = in.get(i, _colID - 1);
			String key = (okey != null) ? okey.toString() : null;
			long code = lookupRCDMap(key);
			out.quickSetValueThreadSafe(i, outputCol, (code >= 0) ? code : Double.NaN);
		}
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		throw new DMLRuntimeException(
			"Recode called with MatrixBlock. Should not happen since Recode is the first " + "encoder in the Stack");
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		throw new DMLRuntimeException(
			"Recode called with MatrixBlock. Should not happen since Recode is the first " + "encoder in the Stack");
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

	private static class RecodePartialBuildTask implements Callable<Object> {

		private final FrameBlock _input;
		private final int _blockSize;
		private final int _startRow;
		private final int _colID;
		private final HashMap<Integer, HashMap<String, Long>> _partialMaps;

		protected RecodePartialBuildTask(FrameBlock input, int colID, int startRow, int blocksize) {
			_input = input;
			_blockSize = blocksize;
			_colID = colID;
			_startRow = startRow;
			_partialMaps = null;
		}

		protected RecodePartialBuildTask(FrameBlock input, int colID, int startRow, int blocksize,
											HashMap<Integer, HashMap<String, Long>> partialMaps) {
			_input = input;
			_blockSize = blocksize;
			_colID = colID;
			_startRow = startRow;
			_partialMaps = partialMaps;
		}


		@Override
		public HashMap<String, Long> call() throws Exception {
			HashMap<String, Long> partialMap = new HashMap<>();
			makeRcdMap(_input, partialMap, _colID, _startRow, _blockSize);
			if(_partialMaps != null){
				synchronized (_partialMaps){
					_partialMaps.put(_startRow, partialMap);
				}
				return null;
			}
			return partialMap;
		}
	}

	private static class RecodeMergePartialBuildTask implements Callable<Object>{
		private final HashMap<Integer, HashMap<String, Long>> _partialMaps;
		private final ColumnEncoderRecode _encoder;

		private RecodeMergePartialBuildTask(ColumnEncoderRecode encoderRecode,
											HashMap<Integer, HashMap<String, Long>> partialMaps) {
			_partialMaps = partialMaps;
			_encoder = encoderRecode;
		}

		@Override
		public Object call() throws Exception {
			HashMap<String, Long> rcdMap = _encoder.getRcdMap();
			_partialMaps.forEach((start_row, map) -> {
				map.forEach((k, v) -> {
					if(!rcdMap.containsKey(k))
						putCode(rcdMap, k);
				});
			});
			_encoder._rcdMap = rcdMap;
			return null;
		}
	}


	private static class ColumnRecodeBuildTask implements Callable<Object> {

		private final ColumnEncoderRecode _encoder;
		private final FrameBlock _input;
		
		protected ColumnRecodeBuildTask(ColumnEncoderRecode encoder, FrameBlock input) {
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
