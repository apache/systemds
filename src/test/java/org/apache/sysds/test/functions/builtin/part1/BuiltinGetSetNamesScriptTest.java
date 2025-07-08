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

import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import static org.junit.Assert.fail;
import static org.junit.Assert.assertArrayEquals;

public class BuiltinGetSetNamesScriptTest extends AutomatedTestBase {

    private static final Log LOG = LogFactory.getLog(BuiltinGetSetNamesScriptTest.class);

    private static final String TEST_NAME = "BuiltinGetSetNamesTest";
    private static final String TEST_DIR = "functions/builtin/part1/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinGetSetNamesScriptTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"N"}));
    }

    @Test
    public void testSetNamesAndGetNames() {
        TestConfiguration config = getTestConfiguration(TEST_NAME);
        loadTestConfiguration(config);

        fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
        programArgs = new String[] { "-args", output("N") };

        runTest(true, false, null, -1);

        String actualOutputPath = output("N");
        String actualContent;

        try {
            actualContent = new String(Files.readAllBytes(Paths.get(actualOutputPath))).trim();
            String[] actualNames = actualContent.split(",");

            String[] expectedNames = new String[]{"name", "age"};

            assertArrayEquals("Column names mismatch.", expectedNames, actualNames);

        } catch (IOException e) {
            LOG.error("Failed to read test files: " + e.getMessage(), e);
            fail("Failed to read test files: " + e.getMessage());
        }
    }
}

