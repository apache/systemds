package org.apache.sysds.test.component.frame.transform;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.EncodeBuildCache;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderType;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

@RunWith(value = Parameterized.class)
public class TransformEncodeCacheUnitTest {
    protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheTestSingleCol.class.getName());

    private final FrameBlock data;
    private final int k;
    private final List<String> specs;
    private final EncoderType encoderType;

    public TransformEncodeCacheUnitTest(FrameBlock data, int k, List<String> specs, EncoderType encoderType) {
        this.data = data;
        this.k = k;
        this.specs = specs;
        this.encoderType = encoderType;
    }

    @BeforeClass
    public static void setUp() {
        FrameBlock setUpData = TestUtils.generateRandomFrameBlock(10, new Types.ValueType[]{Types.ValueType.FP32}, 231);

        MultiColumnEncoder encoder = EncoderFactory.createEncoder("{recode:[C1]}", setUpData.getColumnNames(), setUpData.getNumColumns(), null);
        try {
            long duration = measureEncodeTime(encoder, setUpData, 1);
            LOG.info("Setup took " + duration/1_000_000.0 + " milliseconds");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Parameters
    public static Collection<Object[]> data() {
        final ArrayList<Object[]> tests = new ArrayList<>();
        final int k = 1;
        List<FrameBlock> testData = Arrays.asList(
                TestUtils.generateRandomFrameBlock(10, new Types.ValueType[]{Types.ValueType.STRING, Types.ValueType.STRING, }, 231),
                TestUtils.generateRandomFrameBlock(10, new Types.ValueType[]{Types.ValueType.FP32, Types.ValueType.FP32, }, 231)
                );
        List<List<String>> specLists = Arrays.asList(
                Arrays.asList(
                        "{recode:[C1]}", "{recode:[C2]}"),
                Arrays.asList(
                        "{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}",
                        "{ids:true, bin:[{id:2, method:equi-width, numbins:4}]}")
        );
        List<EncoderType> encoderTypes = Arrays.asList(EncoderType.Recode, EncoderType.Bin);

        for (int index = 0; index < specLists.size(); index++){
            tests.add(new Object[]{testData.get(index), k, specLists.get(index), encoderTypes.get(index)});
        }
        return tests;
    }

    @Test
    public void testEviction(){
        //premise: new element size is bigger than free memory in cache --> need to fill up cache first
        //assertion: least recently used element is removed from cache
        EncodeBuildCache cache = EncodeBuildCache.getEncodeBuildCache();
        cache.setCacheLimit(0.005);
        System.out.println("used Memory: " + cache.get_usedCacheMemory());
    }

    private static long measureEncodeTime(MultiColumnEncoder encoder, FrameBlock data, int k) {
        long startTime = System.nanoTime();
        encoder.encode(data, k);
        long endTime = System.nanoTime();
        return endTime - startTime;
    }
}
