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
import org.apache.sysds.runtime.io.*;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 *
 */
public class TokenizeTest extends AutomatedTestBase
{
    private static final String TEST_NAME1 = "Tokenize";
    private static final String TEST_DIR = "functions/transform/";
    private static final String TEST_CLASS_DIR = TEST_DIR + TokenizeTest.class.getSimpleName() + "/";

    private static final String SPEC = "TokenizeSpec.json";

    //dataset and transform tasks without missing values
    private final static String DATASET 	= "20news/20news_subset_untokenized.csv";  // TODO: use untokenized as input

    @Override
    public void setUp()  {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME1,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
    }

    @Test
    public void testTokenizeSingleNodeCSV() {
        runTokenizeTest(ExecMode.SINGLE_NODE, "csv", false);
    }

    @Test
    public void testTokenizeSparkCSV() {
        runTokenizeTest(ExecMode.SPARK, "csv", false);
    }

    @Test
    public void testTokenizeHybridCSV() {
        runTokenizeTest(ExecMode.HYBRID, "csv", false);
    }

    @Test
    public void testTokenizeParReadSingleNodeCSV() {
        runTokenizeTest(ExecMode.SINGLE_NODE, "csv", true);
    }

    @Test
    public void testTokenizeParReadSparkCSV() {
        runTokenizeTest(ExecMode.SPARK, "csv", true);
    }

    @Test
    public void testTokenizeParReadHybridCSV() {
        runTokenizeTest(ExecMode.HYBRID, "csv", true);
    }

    private void runTokenizeTest(ExecMode rt, String ofmt, boolean parRead )
    {
        //set runtime platform
        ExecMode rtold = rtplatform;
        rtplatform = rt;

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;

        if( !ofmt.equals("csv") )
            throw new RuntimeException("Unsupported test output format");

        try
        {
            getAndLoadTestConfiguration(TEST_NAME1);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
            programArgs = new String[]{"-stats","-args",
                    HOME + "input/" + DATASET, HOME + SPEC, output("R") };

            runTest(true, false, null, -1);

            //read input/output and compare
            FrameReader reader2 = parRead ?
                    new FrameReaderTextCSVParallel( new FileFormatPropertiesCSV() ) :
                    new FrameReaderTextCSV( new FileFormatPropertiesCSV()  );
            FrameBlock fb2 = reader2.readFrameFromHDFS(output("R"), -1L, -1L);
            System.out.println(DataConverter.toString(fb2));
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
