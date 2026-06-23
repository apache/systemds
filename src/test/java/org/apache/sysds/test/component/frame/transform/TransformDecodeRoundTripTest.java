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

import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
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
