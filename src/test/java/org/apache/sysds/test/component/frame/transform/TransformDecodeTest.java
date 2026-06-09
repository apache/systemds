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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Component tests for the transform decoders. These exercise the row-block and parallel decode paths, the sparse and
 * dense dummycode decode paths, the binning source-column offset mapping, and feature-hash column handling end-to-end
 * through an encode followed by decode round trip.
 */
@RunWith(value = Parameterized.class)
public class TransformDecodeTest {
	protected static final Log LOG = LogFactory.getLog(TransformDecodeTest.class.getName());

	private final FrameBlock data;
	private final int k;

	public TransformDecodeTest(FrameBlock data, int k) {
		// name must contain "main" so the parallel decode path reuses the shared thread pool
		Thread.currentThread().setName("main_test_decode");
		Logger.getLogger(CommonThreadPool.class.getName()).setLevel(Level.OFF);
		this.data = data;
		this.k = k;
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int[] threads = new int[] {1, 4};
		try {
			final FrameBlock[] blocks = new FrameBlock[] {
				// single low-cardinality categorical column
				TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT4}, 231),
				// single categorical column with nulls
				TestUtils.generateRandomFrameBlock(64, new ValueType[] {ValueType.UINT4}, 99, 0.2),
				// multi column: dummycode/bin on col1 must offset the trailing passthrough columns
				TestUtils.generateRandomFrameBlock(120,
					new ValueType[] {ValueType.UINT4, ValueType.UINT8, ValueType.FP32}, 17),
				// large enough to split into multiple row blocks in the parallel decode path
				TestUtils.generateRandomFrameBlock(2500, new ValueType[] {ValueType.UINT4}, 7)};

			for(FrameBlock block : blocks)
				for(int k : threads)
					tests.add(new Object[] {block, k});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		return tests;
	}

	@Test
	public void testPassThrough() {
		decodeConsistency("{ids:true}");
	}

	@Test
	public void testRecode() {
		decodeConsistency("{ids:true, recode:[1]}");
	}

	@Test
	public void testDummycode() {
		decodeConsistency("{ids:true, recode:[1], dummycode:[1]}");
	}

	@Test
	public void testBinWidth() {
		decodeConsistency("{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}");
	}

	@Test
	public void testBinHeight() {
		decodeConsistency("{ids:true, bin:[{id:1, method:equi-height, numbins:10}]}");
	}

	@Test
	public void testBinSingleBin() {
		// numbins:1 forces the key==0 branch in the bin decoder
		decodeConsistency("{ids:true, bin:[{id:1, method:equi-width, numbins:1}]}");
	}

	@Test
	public void testHashToDummy() {
		// feature-hash columns carry their domain size as the magic "¿K" metadata value, which the dummycode decoder
		// must parse to reconstruct the one-hot column ranges
		decodeConsistency("{ids:true, hash:[1], K:8, dummycode:[1]}");
	}

	@Test
	public void testHashToDummyDomain1() {
		decodeConsistency("{ids:true, hash:[1], K:1, dummycode:[1]}");
	}

	/**
	 * Encode the data, then decode the encoded matrix in three ways: serial dense, parallel dense, and serial sparse.
	 * All three must produce identical frames. This jointly exercises the parallel block-decode path in
	 * {@link Decoder#decode(MatrixBlock, FrameBlock, int)} and the separate sparse / dense dummycode decode paths.
	 */
	private void decodeConsistency(String spec) {
		try {
			final String[] colnames = data.getColumnNames();
			final MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, data.getNumColumns(), null);
			final MatrixBlock encoded = encoder.encode(data, 1);
			final FrameBlock meta = encoder.getMetaData(null);

			final MatrixBlock dense = forceDense(encoded);
			final MatrixBlock sparse = forceSparse(encoded);

			final FrameBlock reference = decode(spec, colnames, meta, dense, 1);
			final FrameBlock parallel = decode(spec, colnames, meta, dense, k);
			final FrameBlock fromSparse = decode(spec, colnames, meta, sparse, 1);

			assertEquals("decoded rows must match input rows", data.getNumRows(), reference.getNumRows());

			TestUtils.compareFrames(reference, parallel, false);
			TestUtils.compareFrames(reference, fromSparse, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(spec + " : " + e.getMessage());
		}
	}

	private static FrameBlock decode(String spec, String[] colnames, FrameBlock meta, MatrixBlock in, int k) {
		final Decoder decoder = DecoderFactory.createDecoder(spec, colnames, null, meta, in.getNumColumns());
		return decoder.decode(in, new FrameBlock(decoder.getSchema()), k);
	}

	private static MatrixBlock forceDense(MatrixBlock in) {
		final MatrixBlock out = new MatrixBlock();
		out.copy(in);
		if(out.isInSparseFormat())
			out.sparseToDense();
		return out;
	}

	private static MatrixBlock forceSparse(MatrixBlock in) {
		final MatrixBlock out = new MatrixBlock();
		out.copy(in);
		if(!out.isInSparseFormat())
			out.denseToSparse();
		return out;
	}
}
