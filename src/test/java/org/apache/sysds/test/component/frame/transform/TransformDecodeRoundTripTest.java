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

package org.apache.sysds.test.component.frame.transform;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Before;
import org.junit.Test;

/**
 * Exact inverse correctness tests for the transform decoders. Recode and dummycode are lossless category encodings, so a
 * decode of the encoded matrix must reconstruct the original categorical frame. These tests assert exact reconstruction
 * for the dense path, the sparse path, and the parallel path so that the dummycode sparse binary search and the parallel
 * block split are validated against ground truth rather than only against each other.
 */
public class TransformDecodeRoundTripTest {
	protected static final Log LOG = LogFactory.getLog(TransformDecodeRoundTripTest.class.getName());

	@Before
	public void setUp() {
		// name must contain "main" so the parallel decode path reuses the shared thread pool
		Thread.currentThread().setName("main_test_decode");
	}

	private static FrameBlock categoricalFrame() {
		final String[] values = new String[] {
			"apple", "banana", "apple", "cherry", "banana", "date", "apple", "cherry", "date", "banana", "elderberry",
			"apple", "fig", "banana", "cherry", "apple", "date", "fig", "elderberry", "banana"};
		final FrameBlock f = new FrameBlock(new ValueType[] {ValueType.STRING});
		f.ensureAllocatedColumns(values.length);
		for(int i = 0; i < values.length; i++)
			f.set(i, 0, values[i]);
		return f;
	}

	@Test
	public void recodeReconstructsOriginalDense() {
		roundTrip("{ids:true, recode:[1]}", false, 1);
	}

	@Test
	public void recodeReconstructsOriginalSparse() {
		roundTrip("{ids:true, recode:[1]}", true, 1);
	}

	@Test
	public void recodeReconstructsOriginalParallel() {
		roundTrip("{ids:true, recode:[1]}", false, 4);
	}

	@Test
	public void dummycodeReconstructsOriginalDense() {
		roundTrip("{ids:true, recode:[1], dummycode:[1]}", false, 1);
	}

	@Test
	public void dummycodeReconstructsOriginalSparse() {
		// the one-hot encoded matrix is sparse, so this drives the dummycode sparse binary-search decode path
		roundTrip("{ids:true, recode:[1], dummycode:[1]}", true, 1);
	}

	@Test
	public void dummycodeReconstructsOriginalParallel() {
		roundTrip("{ids:true, recode:[1], dummycode:[1]}", false, 4);
	}

	/**
	 * Binning a column while a different column is dummycoded shifts the bin column's source position in the encoded
	 * matrix. The bin decoder must rebuild that source-column mapping from the dummycode domain sizes. This asserts the
	 * dense, sparse, and parallel decode paths agree for that layout (bin output is lossy, so exact reconstruction is
	 * not asserted, only cross-mode consistency and dimensions).
	 */
	@Test
	public void binWithDummycodeOnOtherColumnConsistency() {
		// bin column (1) precedes the dummycode column (2): the bin decoder takes the direct
		// source-column path because no expanded column sits before it
		final FrameBlock original = TestUtils.generateRandomFrameBlock(150,
			new ValueType[] {ValueType.FP32, ValueType.UINT4, ValueType.UINT8}, 4242);
		binConsistency("{ids:true, bin:[{id:1, method:equi-width, numbins:4}], dummycode:[2]}", original);
	}

	/**
	 * Dummycode on an earlier column (1) shifts the bin column (2) to the right in the encoded matrix. The bin decoder
	 * must walk the dummycode domain sizes to recover the bin column's true source position. This drives the
	 * non-magic offset branch of the bin source-column mapping.
	 */
	@Test
	public void binAfterDummycodeOnEarlierColumnConsistency() {
		final FrameBlock original = TestUtils.generateRandomFrameBlock(150,
			new ValueType[] {ValueType.UINT4, ValueType.FP32, ValueType.UINT8}, 4242);
		binConsistency("{ids:true, recode:[1], dummycode:[1], bin:[{id:2, method:equi-width, numbins:4}]}", original);
	}

	/**
	 * Same right-shift as above, but the earlier column is feature-hashed before being dummycoded. The hash domain
	 * size K is stored as a plain integer in the single meta cell, so the bin source-column mapping reads it (instead
	 * of numDistinct) to compute the offset.
	 */
	@Test
	public void binAfterHashDummycodeOnEarlierColumnConsistency() {
		final FrameBlock original = TestUtils.generateRandomFrameBlock(150,
			new ValueType[] {ValueType.UINT4, ValueType.FP32, ValueType.UINT8}, 4242);
		binConsistency("{ids:true, hash:[1], K:6, dummycode:[1], bin:[{id:2, method:equi-width, numbins:4}]}",
			original);
	}

	/**
	 * Encode then decode the dense, parallel and sparse paths and assert they agree. Bin output is lossy, so only
	 * cross-mode consistency and row count are asserted (not exact reconstruction).
	 */
	private void binConsistency(String spec, FrameBlock original) {
		try {
			final String[] colnames = original.getColumnNames();

			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			final FrameBlock meta = encoder.getMetaData(null);

			final MatrixBlock dense = new MatrixBlock();
			dense.copy(encoded);
			if(dense.isInSparseFormat())
				dense.sparseToDense();

			final MatrixBlock sparse = new MatrixBlock();
			sparse.copy(encoded);
			if(!sparse.isInSparseFormat())
				sparse.denseToSparse();

			final FrameBlock reference = decodeOnce(spec, colnames, meta, dense, 1);
			final FrameBlock parallel = decodeOnce(spec, colnames, meta, dense, 4);
			final FrameBlock fromSparse = decodeOnce(spec, colnames, meta, sparse, 1);

			org.junit.Assert.assertEquals(original.getNumRows(), reference.getNumRows());
			TestUtils.compareFrames(reference, parallel, false);
			TestUtils.compareFrames(reference, fromSparse, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * The bin encoder always emits codes &gt;= 1, but the decoder defensively handles a 0 code by mapping it to the
	 * first bin's lower boundary. Inject a 0 into an otherwise validly encoded matrix to exercise that branch.
	 */
	@Test
	public void binDecodeZeroCodeUsesFirstBinBoundary() {
		final String spec = "{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}";
		try {
			final FrameBlock original = TestUtils.generateRandomFrameBlock(50, new ValueType[] {ValueType.FP32}, 13);
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			if(encoded.isInSparseFormat())
				encoded.sparseToDense();
			final FrameBlock meta = encoder.getMetaData(null);

			encoded.set(0, 0, 0); // force a 0 bin code

			final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());
			final FrameBlock decoded = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), 1);

			final double first = Double.parseDouble(decoded.get(0, 0).toString());
			final double second = Double.parseDouble(decoded.get(1, 0).toString());
			// the 0-coded row decodes to the first bin lower bound, which is <= any properly binned center
			org.junit.Assert.assertTrue("0-code must map to the lowest bin boundary", first <= second);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * Spark broadcasts the decoder to executors via Java serialization without re-running initMetaData, so the
	 * decoder must round-trip all of its decode state through writeExternal/readExternal. Decode with a freshly
	 * deserialized decoder and assert it matches the in-memory decode. Covers plain bin and bin-with-dummycode
	 * (the latter exercises the serialized _srcCols/_dcCols source-column mapping).
	 */
	@Test
	public void binDecoderSurvivesSerialization() {
		final FrameBlock original = TestUtils.generateRandomFrameBlock(80, new ValueType[] {ValueType.FP32}, 21);
		serializeRoundTrip("{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}", original);
	}

	@Test
	public void binWithDummycodeDecoderSurvivesSerialization() {
		final FrameBlock original = TestUtils.generateRandomFrameBlock(80,
			new ValueType[] {ValueType.UINT4, ValueType.FP32}, 21);
		serializeRoundTrip("{ids:true, recode:[1], dummycode:[1], bin:[{id:2, method:equi-width, numbins:4}]}",
			original);
	}

	private void serializeRoundTrip(String spec, FrameBlock original) {
		try {
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			if(encoded.isInSparseFormat())
				encoded.sparseToDense();
			final FrameBlock meta = encoder.getMetaData(null);

			final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());
			final FrameBlock expected = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), 1);

			final Decoder restored = serializeDeserialize(decoder);
			final FrameBlock actual = restored.decode(encoded, new FrameBlock(restored.getSchema()), 1);

			TestUtils.compareFrames(expected, actual, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	private static Decoder serializeDeserialize(Decoder decoder) throws Exception {
		final ByteArrayOutputStream bos = new ByteArrayOutputStream();
		try(ObjectOutputStream oos = new ObjectOutputStream(bos)) {
			oos.writeObject(decoder);
		}
		try(ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bos.toByteArray()))) {
			return (Decoder) ois.readObject();
		}
	}

	/**
	 * Feature hashing is non-invertible, so the decode contract for a hash column that is NOT dummycoded is that the
	 * encoded bucket code passes through unchanged. Regression test: a hash-only column must not be dropped from the
	 * decoded frame (it previously was, because hash columns were excluded from passthrough).
	 */
	@Test
	public void hashWithoutDummycodeDecodesToBucketCode() {
		final String spec = "{ids:true, hash:[1], K:8}";
		try {
			final FrameBlock original = categoricalFrame();
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			if(encoded.isInSparseFormat())
				encoded.sparseToDense();
			final FrameBlock meta = encoder.getMetaData(null);

			final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());
			final FrameBlock decoded = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), 1);

			org.junit.Assert.assertEquals(1, decoded.getNumColumns());
			for(int i = 0; i < original.getNumRows(); i++) {
				final Object v = decoded.get(i, 0);
				org.junit.Assert.assertNotNull("hash column must survive decode at row " + i, v);
				org.junit.Assert.assertEquals("hash bucket code must pass through at row " + i, encoded.get(i, 0),
					Double.parseDouble(v.toString()), 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * A corrupt recode meta entry (no token/code separator) must surface as a {@link DMLRuntimeException} during
	 * meta-data initialization rather than a raw parsing exception, so callers get an actionable error. Covers the
	 * defensive try/catch added around the recode-map reconstruction.
	 */
	@Test
	public void recodeInitMetaDataRejectsCorruptEntry() {
		final String spec = "{ids:true, recode:[1]}";
		try {
			final FrameBlock original = categoricalFrame();
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			encoder.encode(original, 1);
			final FrameBlock meta = encoder.getMetaData(null);
			// overwrite the first recode entry with a value lacking the token/code separator
			meta.set(0, 0, "corrupt-entry-without-separator");

			try {
				DecoderFactory.createDecoder(spec, colnames, null, meta, original.getNumColumns());
				fail("expected a corrupt recode entry to be rejected");
			}
			catch(DMLRuntimeException expected) {
				assertTrue("error should identify the recode map reinitialization, got: " + messageChain(expected),
					messageChain(expected).contains("recode map"));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * Federated transform-decode slices a global decoder per worker via {@link Decoder#updateIndexRanges} and
	 * {@link Decoder#subRangeDecoder}. For a single worker covering the whole matrix, the dummycode expansion must
	 * collapse the encoded column count down to the decoded column count, and the resulting sub-range decoder must
	 * reproduce the global decode exactly. Exercises the dummycode index-range and sub-range mapping.
	 */
	@Test
	public void dummycodeSubRangeFullRangeMatchesGlobalDecode() {
		final String spec = "{ids:true, recode:[1], dummycode:[1]}";
		try {
			final FrameBlock original = TestUtils.generateRandomFrameBlock(60,
				new ValueType[] {ValueType.UINT4, ValueType.FP32}, 91);
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			if(encoded.isInSparseFormat())
				encoded.sparseToDense();
			final FrameBlock meta = encoder.getMetaData(null);

			final Decoder global = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());
			final FrameBlock full = global.decode(encoded, new FrameBlock(global.getSchema()), 1);

			// single worker covering the whole matrix: map encoded column range to decoded column range
			final long[] beginDims = {0, 0};
			final long[] endDims = {encoded.getNumRows(), encoded.getNumColumns()};
			global.updateIndexRanges(beginDims, endDims);

			org.junit.Assert.assertEquals("begin column must stay at 0", 0, beginDims[1]);
			org.junit.Assert.assertEquals("dummycode expansion must collapse to the decoded column count",
				full.getNumColumns(), (int) endDims[1]);

			final Decoder sub = global.subRangeDecoder(1, (int) endDims[1] + 1, 0);
			final FrameBlock subDecoded = sub.decode(encoded, new FrameBlock(sub.getSchema()), 1);
			TestUtils.compareFrames(full, subDecoded, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * A federated worker holding only the columns after a dummycoded column must shift its index range left by the
	 * dummycode expansion and receive a sub-range decoder containing just the trailing pass-through columns (the
	 * dummycode and recode decoders drop out). Mirrors the {@code updateIndexRanges} + {@code subRangeDecoder} call
	 * sequence in federated transform-decode, covering the index-range shift for a fully-preceding dummycode column and
	 * the empty sub-range branch.
	 */
	@Test
	public void dummycodeSubRangeExcludingDummycodedColumnKeepsRemaining() {
		final String spec = "{ids:true, recode:[1], dummycode:[1]}";
		try {
			final FrameBlock original = TestUtils.generateRandomFrameBlock(40,
				new ValueType[] {ValueType.UINT4, ValueType.FP32, ValueType.FP32}, 73);
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			if(encoded.isInSparseFormat())
				encoded.sparseToDense();
			final FrameBlock meta = encoder.getMetaData(null);

			final Decoder global = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());

			// the dummycode column expands to (encodedCols - 2) one-hot columns; a worker owning only the two trailing
			// pass-through columns starts after that expanded block in encoded column space
			final int dcWidth = encoded.getNumColumns() - 2;
			final long[] beginDims = {0, dcWidth};
			final long[] endDims = {encoded.getNumRows(), dcWidth + 2};
			final int colStartBefore = (int) beginDims[1];
			global.updateIndexRanges(beginDims, endDims);

			// after collapsing the preceding dummycode expansion, the worker maps to decoded columns 2..3
			org.junit.Assert.assertEquals(1, beginDims[1]);
			org.junit.Assert.assertEquals(3, endDims[1]);

			final Decoder sub = global.subRangeDecoder((int) beginDims[1] + 1, (int) endDims[1] + 1, colStartBefore);
			org.junit.Assert.assertNotNull("pass-through columns must still yield a decoder", sub);
			org.junit.Assert.assertEquals("only the two trailing pass-through columns remain", 2,
				sub.getSchema().length);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * Two recode columns with different domain sizes leave trailing empty (null) cells in the shorter column's
	 * recode-map column. Reconstructing that map must stop at the first null rather than read past it. Recode is
	 * lossless, so the decode must reconstruct the original frame exactly.
	 */
	@Test
	public void recodeMultiColumnWithTrailingNullMapEntries() {
		final String spec = "{ids:true, recode:[1, 2]}";
		try {
			final FrameBlock original = new FrameBlock(new ValueType[] {ValueType.STRING, ValueType.STRING});
			final String[] high = {"a", "b", "c", "d", "e", "f", "g", "h"};
			final String[] low = {"x", "y"};
			final int n = 16;
			original.ensureAllocatedColumns(n);
			for(int i = 0; i < n; i++) {
				original.set(i, 0, high[i % high.length]);
				original.set(i, 1, low[i % low.length]);
			}
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			if(encoded.isInSparseFormat())
				encoded.sparseToDense();
			final FrameBlock meta = encoder.getMetaData(null);

			final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());
			final FrameBlock decoded = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), 1);
			TestUtils.compareFrames(original, decoded, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	/**
	 * The parallel decode path runs per-row-block decode tasks on a thread pool; a failure inside a worker must not be
	 * swallowed but resurface as an unchecked exception to the caller. Feeding a matrix with far fewer columns than the
	 * decoder expects forces an out-of-range access in a worker, which the parallel wrapper must propagate.
	 */
	@Test
	public void parallelDecodeWrapsWorkerException() {
		final String spec = "{ids:true, recode:[1], dummycode:[1]}";
		try {
			final FrameBlock original = categoricalFrame();
			final String[] colnames = original.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			final MatrixBlock encoded = encoder.encode(original, 1);
			final FrameBlock meta = encoder.getMetaData(null);
			final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());

			// far fewer columns than the dummycode decoder reads -> a parallel worker accesses out of range
			final MatrixBlock broken = new MatrixBlock(2, 1, false);
			broken.allocateDenseBlock();
			try {
				decoder.decode(broken, new FrameBlock(decoder.getSchema()), 4);
				fail("expected the parallel decode wrapper to propagate the worker failure");
			}
			catch(DMLRuntimeException expected) {
				assertNotNull("parallel decode wrapper must retain the worker exception as cause",
					expected.getCause());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	private static String messageChain(Throwable t) {
		final StringBuilder sb = new StringBuilder();
		for(Throwable c = t; c != null; c = c.getCause())
			sb.append(c.getMessage()).append('\n');
		return sb.toString();
	}

	private static FrameBlock decodeOnce(String spec, String[] colnames, FrameBlock meta, MatrixBlock in, int k) {
		final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, in.getNumColumns());
		return decoder.decode(in, new FrameBlock(decoder.getSchema()), k);
	}

	private void roundTrip(String spec, boolean sparse, int k) {
		try {
			final FrameBlock original = categoricalFrame();
			final String[] colnames = original.getColumnNames();

			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, original.getNumColumns(),
				null);
			MatrixBlock encoded = encoder.encode(original, 1);
			final FrameBlock meta = encoder.getMetaData(null);

			if(sparse && !encoded.isInSparseFormat())
				encoded.denseToSparse();
			else if(!sparse && encoded.isInSparseFormat())
				encoded.sparseToDense();

			final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, encoded.getNumColumns());
			final FrameBlock decoded = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), k);

			TestUtils.compareFrames(original, decoded, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " (sparse=" + sparse + ", k=" + k + ") : " + e.getMessage());
		}
	}
}
