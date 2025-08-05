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

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.common.Types.ExecMode;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class BuiltinGetSetNamesScriptTest extends AutomatedTestBase {
    private static final String TEST_NAME = "BuiltinGetSetNamesTest";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinGetSetNamesScriptTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
        setExecMode(ExecMode.SINGLE_NODE);
    }

    @Test
    public void testGetSetNames() {
        fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
        String tempFilePath = output("B");
        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        programArgs = new String[]{"-args", tempFilePath};
        runTest(true, false, null, -1);
        try (BufferedReader br = new BufferedReader(new FileReader(tempFilePath))) {
            String header = br.readLine();
            if (header == null || !header.equals("ID,Value")) {
                throw new AssertionError("Test failed: Expected header 'ID,Value', but got: " + header);
            }
        } catch (IOException e) {
            throw new AssertionError("Test failed: Unable to read output file: " + e.getMessage());
        }
    }
}