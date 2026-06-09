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
		final String spec = "{ids:true, bin:[{id:1, method:equi-width, numbins:4}], dummycode:[2]}";
		try {
			final FrameBlock original = TestUtils.generateRandomFrameBlock(150,
				new ValueType[] {ValueType.FP32, ValueType.UINT4, ValueType.UINT8}, 4242);
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
