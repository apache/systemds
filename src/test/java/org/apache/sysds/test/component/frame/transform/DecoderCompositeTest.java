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

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderComposite;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Tests for the multi-threaded {@link DecoderComposite#decode(MatrixBlock, FrameBlock, int)} path.
 *
 * <p>
 * The parallel decode partitions over row blocks and runs all sub decoders in order within each block. This is
 * important for the dummycode+recode case: the recode-on-output decoder reads the category indexes written by the
 * preceding dummycode decoder, so running them out of order produces wrong (or null) values. These tests verify the
 * parallel result equals the single-threaded result and reconstructs the original frame, and they also exercise the
 * {@code k <= 1} short-circuit to the sequential path.
 * </p>
 */
public class DecoderCompositeTest {
	protected static final Log LOG = LogFactory.getLog(DecoderCompositeTest.class.getName());

	/** Enough rows that the parallel path forms multiple row blocks (block size is max(rows/k, 1000)). */
	private static final int ROWS = 8000;

	private static FrameBlock categoricalFrame(int rows, int nCol, int nCat, int seed) {
		ValueType[] schema = new ValueType[nCol];
		for(int c = 0; c < nCol; c++)
			schema[c] = ValueType.STRING;
		String[][] data = new String[rows][nCol];
		Random r = new Random(seed);
		for(int i = 0; i < rows; i++)
			for(int c = 0; c < nCol; c++)
				data[i][c] = "v" + r.nextInt(nCat);
		return new FrameBlock(schema, data);
	}

	private static Decoder buildDecoder(FrameBlock data, String spec, MultiColumnEncoder encoder) {
		FrameBlock meta = encoder.getMetaData(new FrameBlock(data.getNumColumns(), ValueType.STRING));
		return DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
	}

	private void runDecode(String spec, int nCol, int nCat) {
		try {
			FrameBlock data = categoricalFrame(ROWS, nCol, nCat, 17);

			MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), null);
			MatrixBlock encoded = encoder.encode(data, 1);

			Decoder decoder = buildDecoder(data, spec, encoder);

			FrameBlock single = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), 1);
			FrameBlock parallel = decoder.decode(encoded, new FrameBlock(decoder.getSchema()), 4);

			// Parallel decode must match the single-threaded decode exactly.
			TestUtils.compareFrames(single, parallel, false);
			// And both must reconstruct the original categorical values.
			TestUtils.compareFrames(data, parallel, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void recodeOnly() {
		runDecode("{recode:[C1,C2,C3]}", 3, 6);
	}

	@Test
	public void dummycodeAndRecode() {
		// dummycode implies recode-on-output: the composite decoder is [Dummycode, Recode-on-output]
		// and the recode step depends on the indexes the dummycode step writes. This is exactly the
		// ordering the parallel fix protects against breaking.
		runDecode("{dummycode:[C1,C2,C3]}", 3, 5);
	}

	@Test
	public void dummycodeAndRecodeSameColumns() {
		// recode and dummycode listed on the same columns -> recoded then dummycoded, decoded in order.
		runDecode("{recode:[C1,C2], dummycode:[C1,C2]}", 2, 4);
	}

	@Test
	public void singleThreadEqualsParallelManyCategories() {
		runDecode("{dummycode:[C1,C2]}", 2, 25);
	}

	@Test
	public void decoderIsComposite() {
		FrameBlock data = categoricalFrame(100, 2, 3, 1);
		String spec = "{recode:[C1], dummycode:[C2]}";
		MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, data.getColumnNames(),
			data.getNumColumns(), null);
		encoder.encode(data, 1);
		Decoder decoder = buildDecoder(data, spec, encoder);
		if(!(decoder instanceof DecoderComposite))
			fail("expected a DecoderComposite but got " + decoder.getClass().getSimpleName());
	}
}
