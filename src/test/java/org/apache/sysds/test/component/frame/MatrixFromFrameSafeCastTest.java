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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.lib.MatrixBlockFromFrame;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Exercises the defensive NaN fallback in {@link MatrixBlockFromFrame} that triggers when a frame contains cells that
 * cannot be parsed into doubles. The fallback is gated behind {@link DMLConfig#FRAME_TO_MATRIX_WARN_CAST}.
 */
public class MatrixFromFrameSafeCastTest {
	protected static final Log LOG = LogFactory.getLog(MatrixFromFrameSafeCastTest.class.getName());

	/** Captures the expected fallback LOG.error so it does not pollute test output. */
	private TestAppender appender;

	private void setWarnCast(boolean enabled) {
		ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.FRAME_TO_MATRIX_WARN_CAST, String.valueOf(enabled));
	}

	@Before
	public void setUp() {
		appender = LoggingUtils.overwrite();
		MatrixBlockFromFrame.WARNED_FOR_FAILED_CAST = false;
		setWarnCast(true);
	}

	@After
	public void tearDown() {
		LoggingUtils.reinsert(appender);
		// restore the strict (default) behavior to avoid leaking into other tests
		setWarnCast(false);
		MatrixBlockFromFrame.WARNED_FOR_FAILED_CAST = false;
	}

	private static final double NA = Double.NaN;

	/** Expected matrix for {@link #mixedFrame()}: parseable cells keep their value, unparseable cells become NaN. */
	private static final double[][] EXPECTED = {{1.0, 4.0}, {NA, 5.0}, {3.0, NA}};

	/**
	 * Build a string frame mixing parseable numbers with values that cannot be cast to double. The non-numeric cells
	 * force the conversion onto the safe-cast path.
	 */
	private static FrameBlock mixedFrame() {
		Array<?> c1 = ArrayFactory.create(new String[] {"1.0", "abc", "3.0"});
		Array<?> c2 = ArrayFactory.create(new String[] {"4.0", "5.0", "xyz"});
		return new FrameBlock(new Array<?>[] {c1, c2});
	}

	@Test
	public void safeCastSingleThread() {
		FrameBlock fb = mixedFrame();
		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);
		compareSafeCast(mb);
	}

	@Test
	public void safeCastParallel() {
		FrameBlock fb = mixedFrame();
		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 4);
		compareSafeCast(mb);
	}

	@Test
	public void safeCastProvidedOutput() {
		FrameBlock fb = mixedFrame();
		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, new MatrixBlock(3, 2, false), 1);
		compareSafeCast(mb);
	}

	@Test
	public void safeCastNonContiguous() {
		FrameBlock fb = mixedFrame();
		MatrixBlock mb = new MatrixBlock(fb.getNumRows(), fb.getNumColumns(), false);
		mb.allocateBlock();
		DenseBlock spy = spy(mb.getDenseBlock());
		when(spy.isContiguous()).thenReturn(false);
		mb.setDenseBlock(spy);

		mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, mb, 1);
		compareSafeCast(mb);
	}

	@Test
	public void safeCastWarnsOnlyOnce() {
		FrameBlock fb = mixedFrame();

		MatrixBlock first = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);
		assertTrue("Conversion should flag that it fell back to NaN casting",
			MatrixBlockFromFrame.WARNED_FOR_FAILED_CAST);
		compareSafeCast(first);

		// second conversion takes the already-warned branch
		MatrixBlock second = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);
		compareSafeCast(second);

		// the fallback warning must be logged exactly once across both conversions
		final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
		long warnings = log.stream()
			.filter(l -> l.getMessage().toString().contains("falling back to NaN on incompatible cells"))
			.count();
		assertEquals(1, warnings);
	}

	@Test
	public void strictThrowsWhenWarnCastDisabled() {
		// default behavior: incompatible cells fail the whole conversion
		setWarnCast(false);
		FrameBlock fb = mixedFrame();

		Exception e = assertThrows(DMLRuntimeException.class,
			() -> MatrixBlockFromFrame.convertToMatrixBlock(fb, 1));
		assertTrue(e.getMessage().contains("Failed to convert FrameBlock to MatrixBlock"));
	}

	@Test
	public void strictThrowsParallelWhenWarnCastDisabled() {
		// default behavior must also fail fast on the multi-threaded path
		setWarnCast(false);
		FrameBlock fb = mixedFrame();

		Exception e = assertThrows(DMLRuntimeException.class,
			() -> MatrixBlockFromFrame.convertToMatrixBlock(fb, 4));
		assertTrue(e.getMessage().contains("Failed to convert FrameBlock to MatrixBlock"));
	}

	@Test
	public void warnCastValidFrameConvertsWithoutFallback() {
		// warn-cast enabled but every cell is parseable: the strict path succeeds and the NaN
		// fallback must never trigger (covers the try-succeeds branch of convert).
		Array<?> c1 = ArrayFactory.create(new String[] {"1.0", "2.0", "3.0"});
		Array<?> c2 = ArrayFactory.create(new String[] {"4.0", "5.0", "6.0"});
		FrameBlock fb = new FrameBlock(new Array<?>[] {c1, c2});

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);

		compare(new double[][] {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}, mb);
		assertFalse("No cells failed to parse, so the fallback must not have been used",
			MatrixBlockFromFrame.WARNED_FOR_FAILED_CAST);

		final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
		long warnings = log.stream()
			.filter(l -> l.getMessage().toString().contains("falling back to NaN on incompatible cells"))
			.count();
		assertEquals(0, warnings);
	}

	@Test
	public void safeCastZeroValues() {
		// zero-valued parseable cells must not contribute to the non-zero count even on the safe-cast
		// path (covers the ': 0' branch of the nnz ternary), while unparseable cells still become NaN.
		Array<?> c1 = ArrayFactory.create(new String[] {"0.0", "abc"});
		Array<?> c2 = ArrayFactory.create(new String[] {"2.0", "0.0"});
		FrameBlock fb = new FrameBlock(new Array<?>[] {c1, c2});

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);

		compare(new double[][] {{0.0, 2.0}, {NA, 0.0}}, mb);
		// non-zeros: 2.0 and the NaN cell count, the two explicit zeros do not
		assertEquals(2, mb.getNonZeros());
	}

	@Test
	public void safeCastAllInvalid() {
		// every cell fails to parse: the whole matrix becomes NaN and each NaN counts as a non-zero
		Array<?> c1 = ArrayFactory.create(new String[] {"abc", "def"});
		Array<?> c2 = ArrayFactory.create(new String[] {"ghi", "jkl"});
		FrameBlock fb = new FrameBlock(new Array<?>[] {c1, c2});

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);

		compare(new double[][] {{NA, NA}, {NA, NA}}, mb);
		assertEquals(4, mb.getNonZeros());
	}

	@Test
	public void privateConstructor() throws Exception {
		Constructor<MatrixBlockFromFrame> c = MatrixBlockFromFrame.class.getDeclaredConstructor();
		assertTrue("Constructor should be private", Modifier.isPrivate(c.getModifiers()));
		c.setAccessible(true);
		c.newInstance();
	}

	/**
	 * Verify that every parseable cell matches its expected value and every unparseable cell became NaN.
	 */
	private static void compareSafeCast(MatrixBlock mb) {
		compare(EXPECTED, mb);
	}

	/**
	 * Verify that the matrix matches the expected values cell by cell, treating NaN cells as expected NaN.
	 */
	private static void compare(double[][] expected, MatrixBlock mb) {
		assertEquals(expected.length, mb.getNumRows());
		assertEquals(expected[0].length, mb.getNumColumns());
		for(int i = 0; i < expected.length; i++)
			for(int j = 0; j < expected[i].length; j++)
				assertEquals("cell (" + i + "," + j + ")", expected[i][j], mb.get(i, j), 0.0);
	}
}
