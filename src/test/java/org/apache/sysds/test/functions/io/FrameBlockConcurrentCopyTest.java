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

package org.apache.sysds.test.functions.io;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.*;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicInteger;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FrameBlockConcurrentCopyTest extends AutomatedTestBase {

	private final static String TEST_NAME = "FrameBlockConcurrentCopyTest";
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameBlockConcurrentCopyTest.class.getSimpleName() + "/";

	private final int _threads;

	public FrameBlockConcurrentCopyTest(int threads) {
		_threads = threads;
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{20}, {50}});
	}

	/**
	 * Boolean Array Consistency: Verifies that concurrent writes of "All True" or "All False" rows do not result in
	 * mixed rows.
	 */
	@Test
	public void testBooleanArrayConsistency() {
		final int COLS = 64;
		final int ITERATIONS = 100;

		for(int iter = 0; iter < ITERATIONS; iter++) {
			ValueType[] schema = UtilFunctions.nCopies(COLS, ValueType.BOOLEAN);
			FrameBlock target = new FrameBlock(schema, 1);
			target.ensureAllocatedColumns(1);

			// all false
			for(int c = 0; c < COLS; c++)
				target.set(0, c, false);

			CountDownLatch startLatch = new CountDownLatch(1);
			CountDownLatch doneLatch = new CountDownLatch(_threads);
			AtomicInteger errors = new AtomicInteger(0);

			for(int t = 0; t < _threads; t++) {
				final int threadId = t;
				new Thread(() -> {
					try {
						startLatch.await();

						// pattern
						boolean pattern = (threadId % 2 == 0);
						ValueType[] subSchema = UtilFunctions.nCopies(COLS, ValueType.BOOLEAN);

						// source blocks
						FrameBlock source = new FrameBlock(subSchema, 1);
						source.ensureAllocatedColumns(1);
						for(int c = 0; c < COLS; c++)
							source.set(0, c, pattern);

						// repeatedly copy to the shared target
						for(int i = 0; i < 50; i++) {
							// change pattern
							for(int c = 0; c < COLS; c++)
								source.set(0, c, !pattern);
							pattern = !pattern;

							target.copy(0, 0, 0, COLS - 1, source);
						}
					}
					catch(Exception e) {
						errors.incrementAndGet();
					}
					finally {
						doneLatch.countDown();
					}
				}).start();
			}

			startLatch.countDown();
			try {
				doneLatch.await();
			}
			catch(InterruptedException e) {
				Assert.fail("Test interrupted: " + e.getMessage());
			}

			Assert.assertEquals("Exceptions occurred inside worker threads", 0, errors.get());

			// The row must be uniform (All True/False), mix row indicates overwriting
			Boolean firstVal = (Boolean) target.get(0, 0);
			for(int c = 1; c < COLS; c++) {
				Boolean val = (Boolean) target.get(0, c);
				if(val != firstVal) {
					Assert.fail(
						"Inconsistent array detected! Found mixed values in what should be an atomic block write.");
				}
			}
		}
	}

	/**
	 * Non-Boolean Type Safety: Verifies that standard types are thread-safe when threads write to disjoint rows.
	 */
	@Test
	public void testSafeTypesNoSync() {
		final int ITERATIONS = 20;
		final int COLS = 5;
		ValueType[] types = {ValueType.BOOLEAN, ValueType.INT64, ValueType.FP64, ValueType.STRING, ValueType.INT32,
			ValueType.FP32, ValueType.UINT4, ValueType.CHARACTER, ValueType.HASH32, ValueType.HASH64};

		for(ValueType type : types) {
			for(int iter = 0; iter < ITERATIONS; iter++) {
				// 1 row per threat to check disjoint writing
				ValueType[] schema = UtilFunctions.nCopies(COLS, type);
				FrameBlock target = new FrameBlock(schema, _threads);
				target.ensureAllocatedColumns(_threads);

				CyclicBarrier barrier = new CyclicBarrier(_threads);
				AtomicInteger errors = new AtomicInteger(0);
				Thread[] threads = new Thread[_threads];

				for(int t = 0; t < _threads; t++) {
					final int threadId = t;
					threads[t] = new Thread(() -> {
						try {
							barrier.await();

							FrameBlock source = new FrameBlock(schema, 1);
							source.ensureAllocatedColumns(1);
							// value based on threadID
							for(int c = 0; c < COLS; c++) {
								initializeCell(source, 0, c, type, threadId + c);
							}
							// Copy source into target at specific row index
							target.copy(threadId, threadId, 0, COLS - 1, source);

						}
						catch(Exception e) {
							errors.incrementAndGet();
						}
					});
					threads[t].start();
				}

				try {
					for(Thread t : threads)
						t.join();
				}
				catch(InterruptedException e) {
					Assert.fail("Test interrupted: " + e.getMessage());
				}

				Assert.assertEquals("Thread errors detected for type " + type, 0, errors.get());

				for(int r = 0; r < _threads; r++) {
					for(int c = 0; c < COLS; c++) {
						Object val = target.get(r, c);
						verifyCell(val, type, r + c);
					}
				}
			}
		}
	}

	/**
	 * Test 3: High-Contention Bit Packing Stress Test Race-condition test Threads focus rows within shared column
	 */
	@Test
	public void testBooleanBitPacking() {
		for(int iter = 0; iter < 10; iter++) {
			final int ROWS = 1024;
			final int STRESS_LOOPS = 2000;

			ValueType[] schema = new ValueType[] {ValueType.BOOLEAN};
			FrameBlock target = new FrameBlock(schema, ROWS);
			target.ensureAllocatedColumns(ROWS);

			//all false
			for(int r = 0; r < ROWS; r++)
				target.set(r, 0, false);

			// pre-allocate source blocks
			FrameBlock[] sources = new FrameBlock[ROWS];
			for(int i = 0; i < ROWS; i++) {
				sources[i] = new FrameBlock(schema, 1);
				sources[i].ensureAllocatedColumns(1);
				sources[i].set(0, 0, true);
			}

			CyclicBarrier barrier = new CyclicBarrier(ROWS);
			AtomicInteger exceptions = new AtomicInteger(0);
			Thread[] threads = new Thread[ROWS];

			for(int i = 0; i < ROWS; i++) {
				final int rowIndex = i;
				threads[i] = new Thread(() -> {
					try {
						barrier.await();

						// try to maximize overlapp
						for(int k = 0; k < STRESS_LOOPS; k++) {
							target.copy(rowIndex, rowIndex, 0, 0, sources[rowIndex]);
						}
					}
					catch(Exception e) {
						exceptions.incrementAndGet();
					}
				});
				threads[i].start();
			}

			try {
				for(Thread t : threads)
					t.join();
			}
			catch(InterruptedException e) {
				Assert.fail("Test interrupted: " + e.getMessage());
			}

			Assert.assertEquals("Exceptions occurred in threads", 0, exceptions.get());

			// row should be TRUE, any FALSE value means update was overwritten by another thread
			int trueCount = 0;
			for(int r = 0; r < ROWS; r++) {
				Object val = target.get(r, 0);
				if(val != null && (Boolean) val)
					trueCount++;
			}

			Assert.assertEquals("Race condition. Lost updates in iteration " + iter, ROWS, trueCount);
		}
	}

	/**
	 * Test 4: OptionalArray Locking Test
	 */
	@Test
	public void testOptionalArray() {
		for(int i = 0; i < 10; i++) {
			System.out.println("--- RUN " + (i + 1) + " ---");
			runOptional();
		}
	}

	@SuppressWarnings("unchecked")
	private void runOptional() {
		int rows = 50000;
		ValueType type = ValueType.FP64;
		FrameBlock target = new FrameBlock(new ValueType[] {type}, rows);
		target.ensureAllocatedColumns(rows);

		// target (optional array)
		target.setColumn(0, ArrayFactory.allocateOptional(type, rows));

		// source 1 (non-null)
		FrameBlock sourceValid = new FrameBlock(new ValueType[] {type}, rows);
		sourceValid.ensureAllocatedColumns(rows);
		Array<?> sCol1 = ArrayFactory.allocateOptional(type, rows);
		((OptionalArray<Double>) sCol1).fill(1.0d);
		sourceValid.setColumn(0, sCol1);

		// source 2 (all null)
		FrameBlock sourceNull = new FrameBlock(new ValueType[] {type}, rows);
		sourceNull.ensureAllocatedColumns(rows);
		Array<?> sCol2 = ArrayFactory.allocateOptional(type, rows);
		((OptionalArray<Double>) sCol2).fill((Double) null);
		sourceNull.setColumn(0, sCol2);

		CyclicBarrier barrier = new CyclicBarrier(2);
		AtomicInteger errors = new AtomicInteger(0);

		Thread t1 = new Thread(() -> {
			try {
				barrier.await();
				target.copy(1, rows - 1, 0, 0, sourceValid);
			}
			catch(Exception e) {
				e.printStackTrace();
				errors.incrementAndGet();
			}
		});

		Thread t2 = new Thread(() -> {
			try {
				barrier.await();
				target.copy(1, rows - 1, 0, 0, sourceNull);
			}
			catch(Exception e) {
				e.printStackTrace();
				errors.incrementAndGet();
			}
		});

		t1.start();
		t2.start();
		try {
			t1.join();
			t2.join();
		}
		catch(InterruptedException e) {
		}

		int validCount = 0;
		int nullCount = 0;

		for(int r = 1; r < rows; r++) {
			if(target.get(r, 0) == null)
				nullCount++;
			else
				validCount++;
		}

		System.out.println("Valid: " + validCount + ", Null: " + nullCount);

		if(validCount > 0 && nullCount > 0) {
			Assert.fail(
				"Race condition OptionalArray:\n" + "Thread 1 (non-null) and Thread 2 (null) mixed instructions!\n" +
					"Result: " + validCount + " valid, " + nullCount + " null.");
		}
	}

	/**
	 * Test 5: BitSetArray Locking Test
	 */
	@Test
	public void testBitSetArray() {
		for(int i = 0; i < 20; i++) {
			System.out.println("--- BitSet RUN " + (i + 1) + " ---");
			runBitSet();
		}
	}

	private void runBitSet() {
		int rows = 50000;
		ValueType type = ValueType.BOOLEAN;
		FrameBlock target = new FrameBlock(new ValueType[] {type}, rows);
		target.ensureAllocatedColumns(rows);
		// initialize false
		target.setColumn(0, ArrayFactory.allocate(type, rows));

		// Source 1 (all true)
		FrameBlock sourceTrue = new FrameBlock(new ValueType[] {type}, rows);
		sourceTrue.ensureAllocatedColumns(rows);
		Array<?> sCol1 = ArrayFactory.allocate(type, rows);
		// Fill with TRUE
		for(int k = 0; k < rows; k++)
			sCol1.set(k, String.valueOf(true));
		sourceTrue.setColumn(0, sCol1);

		// Source 2 (all false)
		FrameBlock sourceFalse = new FrameBlock(new ValueType[] {type}, rows);
		sourceFalse.ensureAllocatedColumns(rows);
		Array<?> sCol2 = ArrayFactory.allocate(type, rows);
		// fill with false
		for(int k = 0; k < rows; k++)
			sCol2.set(k, String.valueOf(false));
		sourceFalse.setColumn(0, sCol2);

		CyclicBarrier barrier = new CyclicBarrier(2);
		AtomicInteger errors = new AtomicInteger(0);

		Thread t1 = new Thread(() -> {
			try {
				barrier.await();
				target.copy(1, rows - 1, 0, 0, sourceTrue);
			}
			catch(Exception e) {
				e.printStackTrace();
				errors.incrementAndGet();
			}
		});

		Thread t2 = new Thread(() -> {
			try {
				barrier.await();
				target.copy(1, rows - 1, 0, 0, sourceFalse);
			}
			catch(Exception e) {
				e.printStackTrace();
				errors.incrementAndGet();
			}
		});

		t1.start();
		t2.start();
		try {
			t1.join();
			t2.join();
		}
		catch(InterruptedException e) {
		}

		// Verification
		int trueCount = 0;
		int falseCount = 0;

		for(int r = 3; r < rows; r++) {
			Boolean val = (Boolean) target.get(r, 0);
			if(val != null && val)
				trueCount++;
			else
				falseCount++;
		}

		System.out.println("True: " + trueCount + ", False: " + falseCount);

		if(trueCount > 0 && falseCount > 0) {
			Assert.fail(
				"Race condition in BitSetArray:\n" + "Thread True and Thread False mixed instructions!\n" + "Result: " +
					trueCount + " True, " + falseCount + " False.");
		}
	}

	private void initializeCell(FrameBlock fb, int row, int col, ValueType type, int seed) {
		switch(type) {
			case BOOLEAN:
				fb.set(row, col, (seed % 2 == 0));
				break;
			case INT32:
				fb.set(row, col, seed);
				break;
			case INT64:
				fb.set(row, col, (long) seed);
				break;
			case FP32:
				fb.set(row, col, (float) seed);
				break;
			case FP64:
				fb.set(row, col, (double) seed);
				break;
			case UINT8:
				fb.set(row, col, (int) (seed % 127));
				break;
			case UINT4:
				fb.set(row, col, (int) (seed % 15));
				break;
			case CHARACTER:
				fb.set(row, col, (char) ('a' + (seed % 26)));
				break;
			case STRING:
				fb.set(row, col, "v" + seed);
				break;
			case HASH32:
				fb.set(row, col, seed);
				break;
			case HASH64:
				fb.set(row, col, (long) seed);
				break;
			default:
				fb.set(row, col, String.valueOf(seed));
		}
	}

	private void verifyCell(Object val, ValueType type, int expectedSeed) {
		Assert.assertNotNull("Value should not be null", val);
		switch(type) {
			case INT32:
				Assert.assertEquals((int) expectedSeed, ((Integer) val).intValue());
				break;
			case INT64:
				Assert.assertEquals((long) expectedSeed, ((Long) val).longValue());
				break;
			case FP32:
				Assert.assertEquals((float) expectedSeed, ((Float) val).floatValue(), 0.0001);
				break;
			case FP64:
				Assert.assertEquals((double) expectedSeed, ((Double) val).doubleValue(), 0.0001);
				break;
			case UINT8:
				Assert.assertEquals((int) (expectedSeed % 127), ((Integer) val).intValue());
				break;
			case CHARACTER:
				Assert.assertEquals((char) ('a' + (expectedSeed % 26)), ((Character) val).charValue());
				break;
			case STRING:
				Assert.assertEquals("v" + expectedSeed, val);
				break;
			case BOOLEAN:
				Assert.assertEquals((expectedSeed % 2 == 0), val);
				break;
			default:
				break;
		}
	}
}
