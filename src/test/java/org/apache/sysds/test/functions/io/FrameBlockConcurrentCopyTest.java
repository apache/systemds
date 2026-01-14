package org.apache.sysds.test.functions.io;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
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
        return Arrays.asList(new Object[][] {
                {20}, {50}
        });
    }

    /**
     * Boolean Array Consistency: Verifies that concurrent writes of "All True" or "All False"
     * rows do not result in mixed rows.
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
            for(int c = 0; c < COLS; c++) target.set(0, c, false);

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
                        for(int c = 0; c < COLS; c++) source.set(0, c, pattern);

                        // repeatedly copy to the shared target
                        for(int i = 0; i < 50; i++) {
                            // change pattern
                            for(int c = 0; c < COLS; c++) source.set(0, c, !pattern);
                            pattern = !pattern;

                            target.copy(0, 0, 0, COLS - 1, source);
                        }
                    } catch (Exception e) {
                        errors.incrementAndGet();
                    } finally {
                        doneLatch.countDown();
                    }
                }).start();
            }

            startLatch.countDown();
            try {
                doneLatch.await();
            } catch (InterruptedException e) {
                Assert.fail("Test interrupted: " + e.getMessage());
            }

            Assert.assertEquals("Exceptions occurred inside worker threads", 0, errors.get());

            // The row must be uniform (All True/False), mix row indicates overwriting
            Boolean firstVal = (Boolean) target.get(0, 0);
            for(int c = 1; c < COLS; c++) {
                Boolean val = (Boolean) target.get(0, c);
                if(val != firstVal) {
                    Assert.fail("Inconsistent array detected! Found mixed values in what should be an atomic block write.");
                }
            }
        }
    }

    /**
     * Non-Boolean Type Safety: Verifies that standard types are thread-safe
     * when threads write to disjoint rows.
     */
    @Test
    public void testNonBooleanTypesNoSync() {
        final int ITERATIONS = 20;
        final int COLS = 5;
        ValueType[] types = {ValueType.INT64, ValueType.FP64, ValueType.STRING};

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

                        } catch (Exception e) {
                            errors.incrementAndGet();
                        }
                    });
                    threads[t].start();
                }

                try {
                    for(Thread t : threads) t.join();
                } catch (InterruptedException e) {
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
     * Test 3: High-Contention Bit Packing Stress Test
     * Race-condition test
     * Threads focus rows within shared column
     */
    @Test
    public void testBooleanBitPacking() {
        for(int iter = 0; iter < 10; iter++) {
            final int ROWS = 1024;
            final int STRESS_LOOPS = 2000;

            ValueType[] schema = new ValueType[]{ValueType.BOOLEAN};
            FrameBlock target = new FrameBlock(schema, ROWS);
            target.ensureAllocatedColumns(ROWS);

            //all false
            for(int r = 0; r < ROWS; r++) target.set(r, 0, false);

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
                        for(int k=0; k<STRESS_LOOPS; k++) {
                            target.copy(rowIndex, rowIndex, 0, 0, sources[rowIndex]);
                        }
                    } catch (Exception e) {
                        exceptions.incrementAndGet();
                    }
                });
                threads[i].start();
            }

            try {
                for(Thread t : threads) t.join();
            } catch (InterruptedException e) {
                Assert.fail("Test interrupted: " + e.getMessage());
            }

            Assert.assertEquals("Exceptions occurred in threads", 0, exceptions.get());

            // row should be TRUE, any FALSE value means update was overwritten by another thread
            int trueCount = 0;
            for(int r = 0; r < ROWS; r++) {
                Object val = target.get(r, 0);
                if(val != null && (Boolean)val) trueCount++;
            }

            Assert.assertEquals("Race condition. Lost updates in iteration " + iter,
                    ROWS, trueCount);
        }
    }

    private void initializeCell(FrameBlock fb, int row, int col, ValueType type, int seed) {
        switch(type) {
            case BOOLEAN: fb.set(row, col, (seed % 2 == 0)); break;
            case INT64:   fb.set(row, col, (long) seed); break;
            case FP32:    fb.set(row, col, (float) seed); break;
            case FP64:    fb.set(row, col, (double) seed); break;
            case STRING:  fb.set(row, col, "value_" + seed); break;
            default:      fb.set(row, col, seed);
        }
    }

    private void verifyCell(Object val, ValueType type, int expectedSeed) {
        Assert.assertNotNull("Value should not be null", val);
        switch(type) {
            case INT64:
                Assert.assertEquals((long)expectedSeed, ((Long)val).longValue());
                break;
            case FP64:
                Assert.assertEquals((double)expectedSeed, (Double) val, 0.0001);
                break;
            case STRING:
                Assert.assertEquals("value_" + expectedSeed, val);
                break;
            default: break;
        }
    }
}