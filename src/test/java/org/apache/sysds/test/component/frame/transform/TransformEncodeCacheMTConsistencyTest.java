package org.apache.sysds.test.component.frame.transform;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.*;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.*;

import static org.junit.Assert.fail;

@RunWith(value = Parameterized.class)
public class TransformEncodeCacheMTConsistencyTest {

    protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheMultiThreadTest.class.getName());

    private final FrameBlock _data;
    private final List<String> _specs;
    protected static LinkedList<EncodeCacheKey> _evicQueue = null;
    protected static Map<EncodeCacheKey, EncodeCacheEntry> _cacheMap = null;

    public TransformEncodeCacheMTConsistencyTest(FrameBlock _data, List<String> _specs) {
        this._data = _data;
        this._specs = _specs;
    }

    @BeforeClass
    public static void setUp() {
        EncodeCacheConfig.useCache(true);
        try {
            long st = System.nanoTime();
            EncodeBuildCache.getEncodeBuildCache();
            long et = System.nanoTime();
            double setUpTime = (et - st)/1_000_000.0;

            _evicQueue = EncodeBuildCache.get_evictionQueue();
            LOG.debug((String.format("Successfully set up cache in %f milliseconds. " +
                    "Size of eviction queue: %d", setUpTime, _evicQueue.size())));

            _cacheMap = EncodeBuildCache.get_cache();

            LOG.debug((String.format("Cache limit: %d", EncodeBuildCache.get_cacheLimit())));

        } catch(DMLRuntimeException e){
            LOG.error("Creation of cache failed:" + e.getMessage());
        }
        EncodeBuildCache.clear();
    }

    @Parameterized.Parameters
    public static Collection<Object[]> testParameters() {
        final ArrayList<Object[]> tests = new ArrayList<>();

        int numColumns = 50;
        int numRows = 10000;
        FrameBlock testData = EncodeCacheTestUtil.generateTestData(numColumns, numRows);

        //create a list of recode specs referring to one distinct column each
        List<String> recodeSpecs = EncodeCacheTestUtil.generateRecodeSpecs(numColumns);

        //create a list of bin specs referring to one distinct column each
        List<String> binSpecs = EncodeCacheTestUtil.generateBinSpecs(numColumns);

        List<List<String>> specLists = Arrays.asList(recodeSpecs, binSpecs);

        //create test cases for each recoder type
        for (List<String> specList : specLists) {
            tests.add(new Object[]{testData, specList});
        }
        return tests;
    }

    @Test
    public void assertThatMatrixBlockIsEqualForAllThreadNumbers() {
        // Assert that the resulting matrix block is equal independent of the number of threads
        try {
            FrameBlock meta = null;
            List<MultiColumnEncoder> encoders = new ArrayList<>();
            for (String spec: _specs) {
                encoders.add(EncoderFactory.createEncoder(spec, _data.getColumnNames(), _data.getNumColumns(), meta));
            }

            final int[] threads = new int[] {2, 4, 8, 3};

            for (MultiColumnEncoder encoder : encoders) {

                MatrixBlock singleThreadResult = encoder.encode(_data, 1);
                for (int k : threads) {
                    MatrixBlock multiThreadResult = encoder.encode(_data, k);
                    EncodeCacheTestUtil.compareMatrixBlocks(singleThreadResult, multiThreadResult);
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
            fail(e.getMessage());
        }
    }
}
