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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryFrameScalarCPInstruction;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Unit tests that drive the get_categorical_mask instruction directly to exercise the defensive code
 * paths (distinct-count prefix in the metadata frame, default column metadata, non id-based specs and
 * the unsupported opcode guard) that the script-level transform tests cannot reach.
 */
public class GetCategoricalMaskInstructionTest {
	protected static final Log LOG = LogFactory.getLog(GetCategoricalMaskInstructionTest.class.getName());

	private static final String MASK_OPCODE = "get_categorical_mask";

	@BeforeClass
	public static void init() throws java.io.IOException {
		CacheableData.initCaching("get_categorical_mask_instruction_test");
	}

	@Test
	public void dummycodeReadsDistinctCountFromMetadataPrefix() {
		// a metadata cell prefixed with '¿' encodes the number of distinct values inline
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"¿3"}});
		MatrixBlock res = run(meta, "{\"ids\": true, \"dummycode\": [1]}");

		assertEquals(1, res.getNumRows());
		assertEquals(3, res.getNumColumns());
		assertArrayEquals(new double[] {1, 1, 1}, res.getDenseBlockValues(), 0.0);
	}

	@Test
	public void dummycodeDefaultMetadataContributesNoColumns() {
		// first column is dummycoded but carries default metadata (no distinct count) -> 0 columns,
		// the trailing pass-through column keeps the output non-empty
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING, ValueType.STRING},
			new String[][] {{"x", "y"}});
		MatrixBlock res = run(meta, "{\"ids\": true, \"dummycode\": [1]}");

		assertEquals(1, res.getNumRows());
		assertEquals(1, res.getNumColumns());
		assertEquals(0.0, res.get(0, 0), 0.0);
	}

	@Test
	public void noMethodAllColumnsPassThrough() {
		// a spec with only "ids" touches no column: every column is a single, non-categorical output
		FrameBlock meta = metaWithDistinct(3, new int[] {0, 0, 0});
		MatrixBlock res = run(meta, "{\"ids\": true}");

		assertMask(res, new double[] {0, 0, 0});
	}

	@Test
	public void recodeInterleavedWithPassThrough() {
		// categorical (recode, 1 col each) interleaved with continuous pass-through columns
		FrameBlock meta = metaWithDistinct(5, new int[] {0, 0, 0, 0, 0});
		MatrixBlock res = run(meta, "{\"ids\": true, \"recode\": [1, 4]}");

		assertMask(res, new double[] {1, 0, 0, 1, 0});
	}

	@Test
	public void leadingPassThroughThenDummycodeOffsets() {
		// the dummycode expansion must start at the correct offset after three continuous columns
		FrameBlock meta = metaWithDistinct(4, new int[] {0, 0, 0, 3});
		MatrixBlock res = run(meta, "{\"ids\": true, \"dummycode\": [4]}");

		assertMask(res, new double[] {0, 0, 0, 1, 1, 1});
	}

	@Test
	public void multipleDummycodeVaryingDistinctCounts() {
		// several dummycoded columns of different widths, all categorical, no pass-through
		FrameBlock meta = metaWithDistinct(3, new int[] {2, 4, 1});
		MatrixBlock res = run(meta, "{\"ids\": true, \"dummycode\": [1, 2, 3]}");

		assertMask(res, new double[] {1, 1, 1, 1, 1, 1, 1});
	}

	@Test
	public void dummycodeAndPassThroughAndRecodeInterleaved() {
		// dummycode(3) | pass-through | recode | dummycode(2): exercises every offset transition
		FrameBlock meta = metaWithDistinct(4, new int[] {3, 0, 0, 2});
		MatrixBlock res = run(meta, "{\"ids\": true, \"recode\": [3], \"dummycode\": [1, 4]}");

		assertMask(res, new double[] {1, 1, 1, 0, 1, 1, 1});
	}

	@Test
	public void recodeAndDummycodeOnSameColumnExpands() {
		// a column listed in both recode and dummycode must expand to its dummycode width, not collapse
		FrameBlock meta = metaWithDistinct(2, new int[] {4, 0});
		MatrixBlock res = run(meta, "{\"ids\": true, \"recode\": [1], \"dummycode\": [1]}");

		assertMask(res, new double[] {1, 1, 1, 1, 0});
	}

	@Test
	public void hashOnlyColumnStaysSingleCategorical() {
		// a hashed-but-not-dummycoded column is a single categorical column; K must not widen it
		FrameBlock meta = metaWithDistinct(3, new int[] {0, 0, 0});
		MatrixBlock res = run(meta, "{\"ids\": true, \"hash\": [2], \"K\": 5}");

		assertMask(res, new double[] {0, 1, 0});
	}

	@Test
	public void hashDummycodeRecodePassThroughMixed() {
		// col1: hash+dummycode -> K=3 (metadata ignored); col2: pass-through; col3: dummycode(9);
		// col4: pass-through; col5: recode. Verifies hashed columns use K while plain dummycode uses
		// the metadata distinct count, with correct offsets across the whole row.
		FrameBlock meta = metaWithDistinct(5, new int[] {0, 0, 9, 0, 0});
		MatrixBlock res = run(meta, "{\"ids\": true, \"recode\": [5], \"dummycode\": [1, 3], \"hash\": [1], \"K\": 3}");

		assertMask(res, new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1});
	}

	@Test
	public void nonIdSpecMissingIdsKeyThrows() {
		// a spec without the "ids" key must be rejected, not silently mis-interpreted
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("non ID based spec", () -> run(meta, "{\"recode\": [1]}"));
	}

	@Test
	public void nonIdSpecIdsFalseThrows() {
		// "ids": false is equally unsupported
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("non ID based spec", () -> run(meta, "{\"ids\": false, \"recode\": [1]}"));
	}

	@Test
	public void unsupportedBinMethodThrows() {
		// bin expands to bin-count columns under dummycode, which the mask does not model
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("unsupported transform method 'bin'",
			() -> run(meta, "{\"ids\": true, \"bin\": [{\"id\": 1, \"method\": \"equi-width\", \"numbins\": 3}]}"));
	}

	@Test
	public void unsupportedWordEmbeddingMethodThrows() {
		// word_embedding maps a column to an embedding vector (many columns), not a single mask entry
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("unsupported transform method 'word_embedding'",
			() -> run(meta, "{\"ids\": true, \"word_embedding\": [1]}"));
	}

	@Test
	public void unsupportedBagOfWordsMethodThrows() {
		// bag_of_words expands to one column per dictionary token
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("unsupported transform method 'bag_of_words'",
			() -> run(meta, "{\"ids\": true, \"bag_of_words\": [1]}"));
	}

	@Test
	public void unsupportedUdfMethodThrows() {
		// udf output arity is user-defined and cannot be inferred from the spec
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("unsupported transform method 'udf'",
			() -> run(meta, "{\"ids\": true, \"udf\": {\"name\": \"f\", \"ids\": [1]}}"));
	}

	@Test
	public void imputeAndOmitAreAccepted() {
		// impute and omit do not change the output column count or categorical flag, so a spec that
		// only adds them on top of a recoded column must still succeed and mark that column categorical
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		MatrixBlock res = run(meta, "{\"ids\": true, \"recode\": [1], \"impute\": [{\"id\": 1, \"method\": \"global_mode\"}], \"omit\": [1]}");

		assertEquals(1, res.getNumRows());
		assertEquals(1, res.getNumColumns());
		assertEquals(1.0, res.get(0, 0), 0.0);
	}

	@Test
	public void malformedSpecWrapsJsonException() {
		// "ids" present but not a boolean makes spec parsing throw a JSONException, which must be
		// wrapped as a DMLRuntimeException rather than propagating raw
		FrameBlock meta = new FrameBlock(new ValueType[] {ValueType.STRING}, new String[][] {{"a"}});
		assertThrowsMessage("was not a boolean", () -> run(meta, "{\"ids\": 5, \"recode\": [1]}"));
	}

	@Test
	public void unsupportedOpcodeThrows() {
		// any frame-scalar binary opcode other than get_categorical_mask must be rejected
		ExecutionContext ec = ExecutionContextFactory.createContext();
		ec.setAutoCreateVars(true);
		ec.setVariable("F", frameObject(new FrameBlock(new ValueType[] {ValueType.STRING},
			new String[][] {{"a"}})));
		assertThrowsMessage("Unsupported operation", () -> maskInstruction("+").processInstruction(ec));
	}

	/** Assert the action throws a DMLRuntimeException whose message chain contains the expected text. */
	private static void assertThrowsMessage(String expected, Runnable action) {
		try {
			action.run();
			fail("Expected DMLRuntimeException containing \"" + expected + "\" but nothing was thrown");
		}
		catch(DMLRuntimeException e) {
			StringBuilder chain = new StringBuilder();
			for(Throwable t = e; t != null; t = t.getCause())
				chain.append(t.getMessage()).append(" | ");
			assertTrue("Exception chain [" + chain + "] should contain \"" + expected + "\"",
				chain.toString().contains(expected));
		}
	}

	/** Assert the mask is a single row equal to the expected values (which also fixes its width). */
	private static void assertMask(MatrixBlock res, double[] expected) {
		assertEquals(1, res.getNumRows());
		assertEquals(expected.length, res.getNumColumns());
		// compare per cell rather than via getDenseBlockValues(): an all-zero mask has nnz == 0 and
		// therefore no materialized dense block
		double[] actual = new double[expected.length];
		for(int i = 0; i < expected.length; i++)
			actual[i] = res.get(0, i);
		assertArrayEquals(expected, actual, 0.0);
	}

	/**
	 * Build a single-row metadata frame of nCol string columns. A positive distinct[i] is written to
	 * that column's metadata as the recode distinct count (the path real transformencode uses), while
	 * a zero leaves the column with default metadata (a continuous / non-dummycoded column).
	 */
	private static FrameBlock metaWithDistinct(int nCol, int[] distinct) {
		ValueType[] schema = new ValueType[nCol];
		String[][] data = new String[1][nCol];
		for(int i = 0; i < nCol; i++) {
			schema[i] = ValueType.STRING;
			data[0][i] = "v";
		}
		FrameBlock fb = new FrameBlock(schema, data);
		for(int i = 0; i < nCol; i++)
			if(distinct[i] > 0)
				fb.setColumnMetadata(i, new ColumnMetadata(distinct[i]));
		return fb;
	}

	private static MatrixBlock run(FrameBlock meta, String spec) {
		ExecutionContext ec = ExecutionContextFactory.createContext();
		ec.setAutoCreateVars(true);
		maskInstruction(MASK_OPCODE).processGetCategorical(ec, meta, new StringObject(spec));
		return ec.getMatrixObject("out").acquireReadAndRelease();
	}

	private static BinaryFrameScalarCPInstruction maskInstruction(String opcode) {
		String in1 = InstructionUtils.concatOperandParts("F", DataType.FRAME.name(), ValueType.STRING.name(), "false");
		String in2 = InstructionUtils.concatOperandParts("spec", DataType.SCALAR.name(), ValueType.STRING.name(), "true");
		String out = InstructionUtils.concatOperandParts("out", DataType.MATRIX.name(), ValueType.FP64.name(), "false");
		String str = InstructionUtils.concatOperands("CP", opcode, in1, in2, out);
		return (BinaryFrameScalarCPInstruction) BinaryCPInstruction.parseInstruction(str);
	}

	private static FrameObject frameObject(FrameBlock fb) {
		MatrixCharacteristics mc = new MatrixCharacteristics(fb.getNumRows(), fb.getNumColumns(), -1, -1);
		FrameObject fo = new FrameObject("F", new MetaDataFormat(mc, FileFormat.BINARY), fb.getSchema());
		fo.acquireModify(fb);
		fo.release();
		return fo;
	}
}
