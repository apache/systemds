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
