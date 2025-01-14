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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.stats.TransformStatistics;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;

import static org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode.constructRecodeMapEntry;
import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class ColumnEncoderBagOfWords extends ColumnEncoder {

	public static int NUM_SAMPLES_MAP_ESTIMATION = 16000;
	private Map<Object, Integer> _tokenDictionary; // switched from int to long to reuse code from RecodeEncoder
	private HashSet<Object> _tokenDictionaryPart = null;
	protected String _seperatorRegex = "\\s+"; // whitespace
	protected boolean _caseSensitive = false;
	protected int[] _nnzPerRow;
	protected long _nnz = 0;
	protected long[] _nnzPartials;
	protected int _defaultNnzCapacity = 64;
	protected double _avgNnzPerRow = 1.0;

	protected ColumnEncoderBagOfWords(int colID) {
		super(colID);
	}

	public ColumnEncoderBagOfWords() {
		super(-1);
	}

	public ColumnEncoderBagOfWords(ColumnEncoderBagOfWords enc) {
		super(enc._colID);
		_nnzPerRow = enc._nnzPerRow != null ? enc._nnzPerRow.clone() : null;
		_tokenDictionary = enc._tokenDictionary;
		_seperatorRegex = enc._seperatorRegex;
		_caseSensitive = enc._caseSensitive;
	}

	public void setTokenDictionary(HashMap<Object, Integer> dict){
		_tokenDictionary = dict;
	}

	public Map<Object, Integer> getTokenDictionary() {
		return _tokenDictionary;
	}

	protected void initNnzPartials(int rows, int numBlocks){
		_nnzPerRow = new int[rows];
		_nnzPartials = new long[numBlocks];
	}

	public double computeNnzEstimate(CacheBlock<?> in, int[] sampleIndices) {
		// estimates the nnz per row for this encoder
		final int max_index = Math.min(ColumnEncoderBagOfWords.NUM_SAMPLES_MAP_ESTIMATION, sampleIndices.length);
		int nnz = 0;
		for (int i = 0; i < max_index; i++) {
			int sind = sampleIndices[i];
			String current = in.getString(sind, _colID - 1);
			if(current != null)
				for(String token : tokenize(current, _caseSensitive, _seperatorRegex))
					if(!token.isEmpty() && _tokenDictionary.containsKey(token))
						nnz++;
		}
		return (double) nnz / max_index;
	}

	public void computeMapSizeEstimate(CacheBlock<?> in, int[] sampleIndices) {
		// Find the frequencies of distinct values in the sample after tokenization
		HashMap<String, Integer> distinctFreq = new HashMap<>();
		long totSize = 0;
		final int max_index = Math.min(ColumnEncoderBagOfWords.NUM_SAMPLES_MAP_ESTIMATION, sampleIndices.length/3);
		int numTokensSample = 0;
		int[] nnzPerRow = new int[max_index];
		for (int i = 0; i < max_index; i++) {
			int sind = sampleIndices[i];
			String current = in.getString(sind, _colID - 1);
			Set<String> tokenSetRow = new HashSet<>();
			if(current != null)
				for(String token : tokenize(current, _caseSensitive, _seperatorRegex))
					if(!token.isEmpty()){
						tokenSetRow.add(token);
						if (distinctFreq.containsKey(token))
							distinctFreq.put(token, distinctFreq.get(token) + 1);
						else {
							distinctFreq.put(token, 1);
							// Maintain total size of the keys
							totSize += (token.length() * 2L + 16); //sizeof(String) = len(chars) + header
						}
						numTokensSample++;
					}
			nnzPerRow[i] = tokenSetRow.size();
		}
		Arrays.sort(nnzPerRow);
		_avgNnzPerRow = (double) Arrays.stream(nnzPerRow).sum() / nnzPerRow.length;
		// default value for HashSets in build phase -> 75% without resize (Division by 0.9 -> is the resize threshold)
		_defaultNnzCapacity = (int) Math.max( nnzPerRow[(int) (nnzPerRow.length*0.75)] / 0.9, 64);
		// we increase the upperbound of the total count estimate by 20%
		double avgSentenceLength = numTokensSample*1.2 / max_index;


		// Estimate total #distincts using Hass and Stokes estimator
		int[] freq = distinctFreq.values().stream().mapToInt(v -> v).toArray();
		_estNumDistincts = SampleEstimatorFactory.distinctCount(freq, (int) (avgSentenceLength*in.getNumRows()),
				numTokensSample, SampleEstimatorFactory.EstimationType.HassAndStokes);

		// Based on a small experimental evaluation:
		// we increase the upperbound of the total count estimate by 2%
		_estNumDistincts = (int) (_estNumDistincts* 1.2);

		// Compute total size estimates for each partial recode map
		// We assume each partial map contains all distinct values and have the same size
		long avgKeySize = totSize / distinctFreq.size();
		long valSize = 16L; //sizeof(Long) = 8 + header
		_avgEntrySize = avgKeySize + valSize;
		_estMetaSize = _estNumDistincts * _avgEntrySize;
	}

	public void computeNnzPerRow(CacheBlock<?> in, int start, int end){
		for (int i = start; i < end; i++) {
			String current = in.getString(i, _colID - 1);
			HashSet<String> distinctTokens = new HashSet<>();
			if(current != null)
				for(String token : tokenize(current, _caseSensitive, _seperatorRegex))
					if(!token.isEmpty() && _tokenDictionary.containsKey(token))
						distinctTokens.add(token);
			_nnzPerRow[i] = distinctTokens.size();
		}
	}

	public static String[] tokenize(String current, boolean caseSensitive, String seperatorRegex) {
		// string builder is faster than regex
		StringBuilder finalString = new StringBuilder();
		for (char c : current.toCharArray()) {
			if (Character.isLetter(c))
				finalString.append(caseSensitive ? c : Character.toLowerCase(c));
			else
				finalString.append(' ');
		}
		return finalString.toString().split(seperatorRegex);
	}

	@Override
	public int getDomainSize(){
		return _tokenDictionary.size();
	}

	@Override
	protected double getCode(CacheBlock<?> in, int row) {
		throw new NotImplementedException();
	}

	@Override
	protected double[] getCodeCol(CacheBlock<?> in, int startInd, int rowEnd, double[] tmp) {
		throw new NotImplementedException();
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.BAG_OF_WORDS;
	}

	public Callable<Object> getBuildTask(CacheBlock<?> in) {
		return new ColumnBagOfWordsBuildTask(this, in);
	}

	@Override
	public void build(CacheBlock<?> in) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		_tokenDictionary = new HashMap<>(_estNumDistincts);
		int i = 1;
		_nnz = 0;
		_nnzPerRow = new int[in.getNumRows()];
		HashSet<String> tokenSetPerRow;
		for (int r = 0; r < in.getNumRows(); r++) {
			// start with a higher default capacity to avoid resizes
			tokenSetPerRow = new HashSet<>(_defaultNnzCapacity);
			String current = in.getString(r, _colID - 1);
			if(current != null)
				for(String token : tokenize(current, _caseSensitive, _seperatorRegex))
					if(!token.isEmpty()){
						tokenSetPerRow.add(token);
						if(!_tokenDictionary.containsKey(token))
							_tokenDictionary.put(token, i++);
					}
			_nnzPerRow[r] = tokenSetPerRow.size();
			_nnz += tokenSetPerRow.size();
		}
		if(DMLScript.STATISTICS)
			TransformStatistics.incBagOfWordsBuildTime(System.nanoTime()-t0);
	}

	@Override
	public Callable<Object> getPartialBuildTask(CacheBlock<?> in, 
		int startRow, int blockSize, HashMap<Integer, Object> ret, int pos) {
		return new BowPartialBuildTask(in, _colID, startRow, blockSize, ret,
			_nnzPerRow, _caseSensitive, _seperatorRegex, _nnzPartials, pos);
	}

	@Override
	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret) {
		_tokenDictionary = new HashMap<>(_estNumDistincts);
		return new BowMergePartialBuildTask(this, ret);
	}

	// Pair class to hold key-value pairs (colId-tokenCount pairs)
	static class Pair {
		int key;
		int value;

		Pair(int key, int value) {
			this.key = key;
			this.value = value;
		}
	}

	@Override
	public void prepareBuildPartial() {
		// ensure allocated partial recode map
		if(_tokenDictionaryPart == null)
			_tokenDictionaryPart = new HashSet<>();
	}


	public HashSet<Object> getPartialTokenDictionary(){
		return _tokenDictionaryPart;
	}

	@Override
	public void buildPartial(FrameBlock in) {
		if(!isApplicable())
			return;
		for (int r = 0; r < in.getNumRows(); r++) {
			String current = in.getString(r, _colID - 1);
			if(current != null)
				for(String token : tokenize(current, _caseSensitive, _seperatorRegex)){
					if(!token.isEmpty())
						_tokenDictionaryPart.add(token);
				}
		}
	}

	protected void applySparse(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;
		mcsr = false; // force CSR for transformencode FIXME
		List<Integer> sparseRowsWZeros = new ArrayList<>();
		for(int r = rowStart; r < getEndIndex(in.getNumRows(), rowStart, blk); r++) {
			if(mcsr) {
				throw new NotImplementedException();
			}
			else { // csr
				HashMap<String, Integer> counter = countTokenAppearances(in, r);
				if(counter.isEmpty())
					sparseRowsWZeros.add(r);
				else {
					SparseBlockCSR csrblock = (SparseBlockCSR) out.getSparseBlock();
					int[] rptr = csrblock.rowPointers();
					// assert that nnz from build is equal to nnz from apply
					Pair[] columnValuePairs = new Pair[_nnzPerRow[r]];
					int i = 0;
					for (Map.Entry<String, Integer> entry : counter.entrySet()) {
						String token = entry.getKey();
						columnValuePairs[i] = new Pair((int) (outputCol + _tokenDictionary.getOrDefault(token, 0) - 1), entry.getValue());
						// if token is not included columnValuePairs[i] is overwritten in the next iteration
						i += _tokenDictionary.containsKey(token) ? 1 : 0;
					}
					// insertion sorts performs better on small arrays
					if(columnValuePairs.length >= 128)
						Arrays.sort(columnValuePairs, Comparator.comparingInt(pair -> pair.key));
					else
						insertionSort(columnValuePairs);
					// Manually fill the column-indexes and values array
					for (i = 0; i < columnValuePairs.length; i++) {
						int index = sparseRowPointerOffset != null ? sparseRowPointerOffset[r] - 1 + i : i;
						index += rptr[r] + _colID -1;
						csrblock.indexes()[index] = columnValuePairs[i].key;
						csrblock.values()[index] = columnValuePairs[i].value;
					}
				}
			}
		}
		if(!sparseRowsWZeros.isEmpty()) {
			addSparseRowsWZeros(sparseRowsWZeros);
		}
	}

	private static void insertionSort(Pair [] arr) {
		for (int i = 1; i < arr.length; i++) {
			Pair current = arr[i];
			int j = i - 1;
			while (j >= 0 && arr[j].key > current.key) {
				arr[j + 1] = arr[j];
				j--;
			}
			arr[j + 1] = current;
		}
	}

	@Override
	protected void applyDense(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk){
		for (int r = rowStart; r < Math.max(in.getNumRows(), rowStart + blk); r++) {
			HashMap<String, Integer> counter = countTokenAppearances(in, r);
			for (Map.Entry<String, Integer> entry : counter.entrySet())
				out.set(r, (int) (outputCol + _tokenDictionary.get(entry.getKey()) - 1), entry.getValue());
		}
	}

	private HashMap<String, Integer> countTokenAppearances(
			CacheBlock<?> in, int r)
	{
		String current = in.getString(r, _colID - 1);
		HashMap<String, Integer> counter = new HashMap<>();
		if(current != null)
			for (String token : tokenize(current, _caseSensitive, _seperatorRegex))
				if (!token.isEmpty() && _tokenDictionary.containsKey(token))
					counter.put(token, counter.getOrDefault(token, 0) + 1);
		return counter;
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		meta.ensureAllocatedColumns(getDomainSize());
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		int rowID = 0;
		StringBuilder sb = new StringBuilder();
		for(Map.Entry<Object, Integer> e : _tokenDictionary.entrySet()) {
			out.set(rowID++, _colID - 1, constructRecodeMapEntry(e.getKey(), e.getValue(), sb));
		}
		return out;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		if(meta != null && meta.getNumRows() > 0) {
			_tokenDictionary = meta.getRecodeMap(_colID - 1);
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);

		out.writeInt(_tokenDictionary == null ? 0 : _tokenDictionary.size());
		if(_tokenDictionary != null)
			for(Map.Entry<Object, Integer> e : _tokenDictionary.entrySet()) {
				System.out.println(e);
				out.writeUTF((String) e.getKey());
				out.writeInt(e.getValue());
			}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		int size = in.readInt();
		_tokenDictionary = new HashMap<>(size * 4 / 3);
		for(int j = 0; j < size; j++) {
			String key = in.readUTF();
			Integer value = in.readInt();
			_tokenDictionary.put(key, value);
		}
	}

	private static class BowPartialBuildTask implements Callable<Object> {

		private final CacheBlock<?> _input;
		private final int _blockSize;
		private final int _startRow;
		private final int _colID;
		private final boolean _caseSensitive;
		private final String _seperator;
		private final HashMap<Integer, Object> _partialMaps;
		private final int[] _nnzPerRow;
		private final long[] _nnzPartials;
		private final int _pos;

		protected BowPartialBuildTask(CacheBlock<?> input, int colID, int startRow,
			int blocksize, HashMap<Integer, Object> partialMaps, int[] nnzPerRow,
			boolean caseSensitive, String seperator, long[] nnzPartials, int pos)
		{
			_input = input;
			_blockSize = blocksize;
			_colID = colID;
			_startRow = startRow;
			_partialMaps = partialMaps;
			_caseSensitive = caseSensitive;
			_seperator = seperator;
			_nnzPerRow = nnzPerRow;
			_nnzPartials = nnzPartials;
			_pos = pos;
		}

		@Override
		public Object call(){
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			int endRow = getEndIndex(_input.getNumRows(), _startRow, _blockSize);
			HashSet<String> tokenSetPartial = new HashSet<>();
			HashSet<String> tokenSetPerRow;
			long nnzPartial = 0;
			for (int r = _startRow; r < endRow; r++) {
				tokenSetPerRow = new HashSet<>(64);
				String current = _input.getString(r, _colID - 1);
				if(current != null)
					for(String token : tokenize(current, _caseSensitive, _seperator))
						if(!token.isEmpty()){
							tokenSetPerRow.add(token);
							tokenSetPartial.add(token);
						}
				_nnzPerRow[r] = tokenSetPerRow.size();
				nnzPartial += tokenSetPerRow.size();
			}
			_nnzPartials[_pos] = nnzPartial;
			synchronized (_partialMaps){
				_partialMaps.put(_startRow, tokenSetPartial);
			}
			if(DMLScript.STATISTICS){
				TransformStatistics.incBagOfWordsBuildTime(System.nanoTime() - t0);
			}
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<Start row: " + _startRow + "; Block size: " + _blockSize + ">";
		}
	}

	private static class BowMergePartialBuildTask implements Callable<Object> {
		private final HashMap<Integer,?> _partialMaps;
		private final ColumnEncoderBagOfWords _encoder;

		private BowMergePartialBuildTask(ColumnEncoderBagOfWords encoderRecode, HashMap<Integer, ?> partialMaps) {
			_partialMaps = partialMaps;
			_encoder = encoderRecode;
		}

		@Override
		public Object call() {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			Map<Object, Integer> tokenDictionary = _encoder._tokenDictionary;
			for(Object tokenSet : _partialMaps.values()){
				( (HashSet<?>) tokenSet).forEach(token -> {
					if(!tokenDictionary.containsKey(token))
						tokenDictionary.put(token, tokenDictionary.size() + 1);
				});
			}
			for (long nnzPartial : _encoder._nnzPartials)
				_encoder._nnz += nnzPartial;
			if(DMLScript.STATISTICS){
				TransformStatistics.incBagOfWordsBuildTime(System.nanoTime() - t0);
			}
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}
	}

	private static class ColumnBagOfWordsBuildTask implements Callable<Object> {

		private final ColumnEncoderBagOfWords _encoder;
		private final CacheBlock<?> _input;

		protected ColumnBagOfWordsBuildTask(ColumnEncoderBagOfWords encoder, CacheBlock<?> input) {
			_encoder = encoder;
			_input = input;
		}

		@Override
		public Void call() {
			_encoder.build(_input);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}
	}
}
