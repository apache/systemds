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

package org.apache.sysds.test.functions.transform;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;


public class TokenizeTest extends AutomatedTestBase  {
    private static final String TEST_DIR = "functions/transform/";
    private static final String TEST_CLASS_DIR = TEST_DIR + TokenizeTest.class.getSimpleName() + "/";

    private static final String TEST_SPLIT_COUNT_LONG = "tokenize/TokenizeSplitCountLong";
    private static final String TEST_NGRAM_POS_LONG = "tokenize/TokenizeNgramPosLong";
    private static final String TEST_NGRAM_POS_WIDE = "tokenize/TokenizeNgramPosWide";
    private static final String TEST_UNI_HASH_WIDE = "tokenize/TokenizeUniHashWide";

    //dataset and transform tasks without missing values
    private final static String DATASET 	= "20news/20news_subset_untokenized.csv";

    @Override
    public void setUp()  {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_SPLIT_COUNT_LONG,
                new TestConfiguration(TEST_CLASS_DIR, TEST_SPLIT_COUNT_LONG, new String[] { "R" }) );
        addTestConfiguration(TEST_NGRAM_POS_LONG,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NGRAM_POS_LONG, new String[] { "R" }) );
        addTestConfiguration(TEST_NGRAM_POS_WIDE,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NGRAM_POS_WIDE, new String[] { "R" }) );
        addTestConfiguration(TEST_UNI_HASH_WIDE,
                new TestConfiguration(TEST_CLASS_DIR, TEST_UNI_HASH_WIDE, new String[] { "R" }) );
    }

    @Test
    public void testTokenizeSingleNodeSplitCountLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_SPLIT_COUNT_LONG,false);
    }

    @Test
    public void testTokenizeSparkSplitCountLong() {

        runTokenizeTest(ExecMode.SPARK, TEST_SPLIT_COUNT_LONG, false);
    }

    @Test
    public void testTokenizeHybridSplitCountLong() {


        runTokenizeTest(ExecMode.HYBRID, TEST_SPLIT_COUNT_LONG, false);
    }

    @Test
    public void testTokenizeParReadSingleNodeSplitCountLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_SPLIT_COUNT_LONG, true);
    }

    @Test
    public void testTokenizeParReadSparkSplitCountLong() {
        runTokenizeTest(ExecMode.SPARK, TEST_SPLIT_COUNT_LONG, true);
    }

    @Test
    public void testTokenizeParReadHybridSplitCountLong() {
        runTokenizeTest(ExecMode.HYBRID, TEST_SPLIT_COUNT_LONG, true);
    }

    @Test
    public void testTokenizeSingleNodeNgramPosLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_NGRAM_POS_LONG,false);
    }

    @Test
    public void testTokenizeSparkNgramPosLong() {
        runTokenizeTest(ExecMode.SPARK, TEST_NGRAM_POS_LONG, false);
    }

    @Test
    public void testTokenizeHybridNgramPosLong() {
        runTokenizeTest(ExecMode.HYBRID, TEST_NGRAM_POS_LONG, false);
    }

    @Test
    public void testTokenizeParReadSingleNodeNgramPosLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_NGRAM_POS_LONG, true);
    }

    @Test
    public void testTokenizeParReadSparkNgramPosLong() {
        runTokenizeTest(ExecMode.SPARK, TEST_NGRAM_POS_LONG, true);
    }

    @Test
    public void testTokenizeParReadHybridNgramPosLong() {
        runTokenizeTest(ExecMode.HYBRID, TEST_NGRAM_POS_LONG, true);
    }

    @Test
    public void testTokenizeSingleNodeNgramPosWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_NGRAM_POS_WIDE,false);
    }

    @Test
    public void testTokenizeSparkNgramPosWide() {
        runTokenizeTest(ExecMode.SPARK, TEST_NGRAM_POS_WIDE, false);
    }

    @Test
    public void testTokenizeHybridNgramPosWide() {
        runTokenizeTest(ExecMode.HYBRID, TEST_NGRAM_POS_WIDE, false);
    }

    @Test
    public void testTokenizeParReadSingleNodeNgramPosWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_NGRAM_POS_WIDE, true);
    }

    @Test
    public void testTokenizeParReadSparkNgramPosWide() {
        runTokenizeTest(ExecMode.SPARK, TEST_NGRAM_POS_WIDE, true);
    }

    @Test
    public void testTokenizeParReadHybridNgramPosWide() {
        runTokenizeTest(ExecMode.HYBRID, TEST_NGRAM_POS_WIDE, true);
    }

    @Test
    public void testTokenizeSingleNodeUniHasWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_UNI_HASH_WIDE,false);
    }

    @Test
    public void testTokenizeSparkUniHasWide() {
        runTokenizeTest(ExecMode.SPARK, TEST_UNI_HASH_WIDE, false);
    }

    @Test
    public void testTokenizeHybridUniHasWide() {
        runTokenizeTest(ExecMode.HYBRID, TEST_UNI_HASH_WIDE, false);
    }

    @Test
    public void testTokenizeParReadSingleNodeUniHasWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TEST_UNI_HASH_WIDE, true);
    }

    @Test
    public void testTokenizeParReadSparkUniHasWide() {
        runTokenizeTest(ExecMode.SPARK, TEST_UNI_HASH_WIDE, true);
    }

    @Test
    public void testTokenizeParReadHybridUniHasWide() {
        runTokenizeTest(ExecMode.HYBRID, TEST_UNI_HASH_WIDE, true);
    }

    private void runTokenizeTest(ExecMode rt, String test_name, boolean parRead )
    {
        //set runtime platform
        ExecMode rtold = rtplatform;
        rtplatform = rt;

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;

        try
        {
            getAndLoadTestConfiguration(test_name);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + test_name + ".dml";
            programArgs = new String[]{"-stats","-args",
                    DATASET_DIR + DATASET, HOME + test_name + ".json", output("R") };

            runTest(true, false, null, -1);
//          TODO add assertion tests
            //read input/output and compare
//            FrameReader reader2 = parRead ?
//                    new FrameReaderTextCSVParallel( new FileFormatPropertiesCSV() ) :
//                    new FrameReaderTextCSV( new FileFormatPropertiesCSV()  );
//            FrameBlock fb2 = reader2.readFrameFromHDFS(output("R"), -1L, -1L);
//            System.out.println(DataConverter.toString(fb2));
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            rtplatform = rtold;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }
}
