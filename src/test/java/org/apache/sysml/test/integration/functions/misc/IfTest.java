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

package org.apache.sysml.test.integration.functions.misc;


import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Test;

public class IfTest extends AutomatedTestBase
{

    private final static String TEST_DIR = "functions/misc/";
    private final static String TEST_NAME1 = "IfTest";
    private final static String TEST_NAME2 = "IfTest2";
    private final static String TEST_NAME3 = "IfTest3";
    private final static String TEST_NAME4 = "IfTest4";
    private final static String TEST_NAME5 = "IfTest5";
    private final static String TEST_NAME6 = "IfTest6";
    private final static String TEST_CLASS_DIR = TEST_DIR + IfTest.class.getSimpleName() + "/";

    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
        addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
        addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
        addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {}));
        addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {}));
        addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {}));
    }

    @Test
    public void testIf() { runTest(TEST_NAME1, 1); }

    @Test
    public void testIfElse() {
        runTest(TEST_NAME2, 1);
        runTest(TEST_NAME2, 2);
    }

    @Test
    public void testIfElif() {
        runTest(TEST_NAME3, 1);
        runTest(TEST_NAME3, 2);
    }

    @Test
    public void testIfElifElse() {
        runTest(TEST_NAME4, 1);
        runTest(TEST_NAME4, 2);
        runTest(TEST_NAME4, 3);
    }

    @Test
    public void testIfElifElif() {
        runTest(TEST_NAME5, 1);
        runTest(TEST_NAME5, 2);
        runTest(TEST_NAME5, 3);
    }

    @Test
    public void testIfElifElifElse() {
        runTest(TEST_NAME6, 1);
        runTest(TEST_NAME6, 2);
        runTest(TEST_NAME6, 3);
        runTest(TEST_NAME6, 4);
    }

    private void runTest( String testName, int val )
    {
        TestConfiguration config = getTestConfiguration(testName);
        loadTestConfiguration(config);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + testName + ".pydml";
        programArgs = new String[]{"-python","-nvargs","val=" + Integer.toString(val)};

        if (val == 1)
            setExpectedStdOut("A");
        else if (val == 2)
            setExpectedStdOut("B");
        else if (val == 3)
            setExpectedStdOut("C");
        else
            setExpectedStdOut("D");

        runTest(true, false, null, -1);
    }
}
