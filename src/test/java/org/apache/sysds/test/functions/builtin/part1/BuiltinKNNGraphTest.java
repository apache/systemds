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

package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
//import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinKNNGraphTest extends AutomatedTestBase {
    private final static String TEST_NAME = "knnGraph";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinKNNGraphTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "PC", "V" }));
    }

    @Test
    public void testKNNGraph() {
        runKNNGraphTest(ExecMode.SINGLE_NODE);
    }

    private void runKNNGraphTest(ExecMode exec_mode) {
        ExecMode platform_old = setExecMode(exec_mode);

        getAndLoadTestConfiguration(TEST_NAME);
        String HOME = SCRIPT_DIR + TEST_DIR;

        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] { "-stats", "-nvargs",
                "in_k=" + Integer.toString(2) };

        // execute tests
        runTest(true, false, null, -1);

        // restore execution mode
        setExecMode(platform_old);
    }

}