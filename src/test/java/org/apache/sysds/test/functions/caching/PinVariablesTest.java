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

package org.apache.sysds.test.functions.caching;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.Data;

import java.util.LinkedList;
import java.util.Queue;
import java.util.List;

public class PinVariablesTest extends AutomatedTestBase {
    private final static String TEST_NAME = "PinVariables";
    private final static String TEST_DIR = "functions/caching/";
    private final static String TEST_CLASS_DIR = TEST_DIR + PinVariablesTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
    }

    @Test
    public void testPinNoLists() {
        createMockDataAndCall(true, false, false);
    }

    @Test
    public void testPinShallowLists() {
        createMockDataAndCall(true, true, false);
    }

    @Test
    public void testPinNestedLists() {
        createMockDataAndCall(true, true, true);
    }

    private void createMockDataAndCall(boolean matrices, boolean list, boolean nestedList) {
        LocalVariableMap vars = new LocalVariableMap();
        List<String> varList = new LinkedList<>();
        Queue<Boolean> varStates = new LinkedList<>();

        if (matrices) {
            MatrixObject mat1 = new MatrixObject(Types.ValueType.FP64, "SomeFile1");
            mat1.enableCleanup(true);
            MatrixObject mat2 = new MatrixObject(Types.ValueType.FP64, "SomeFile2");
            mat2.enableCleanup(true);
            MatrixObject mat3 = new MatrixObject(Types.ValueType.FP64, "SomeFile3");
            mat3.enableCleanup(false);
            vars.put("mat1", mat1);
            vars.put("mat2", mat2);
            vars.put("mat3", mat3);

            varList.add("mat2");
            varList.add("mat3");

            varStates.add(true);
            varStates.add(false);
        }
        if (list) {
            MatrixObject mat4 = new MatrixObject(Types.ValueType.FP64, "SomeFile4");
            mat4.enableCleanup(true);
            MatrixObject mat5 = new MatrixObject(Types.ValueType.FP64, "SomeFile5");
            mat5.enableCleanup(false);
            List<Data> l1_data = new LinkedList<>();
            l1_data.add(mat4);
            l1_data.add(mat5);

            if (nestedList) {
                MatrixObject mat6 = new MatrixObject(Types.ValueType.FP64, "SomeFile6");
                mat4.enableCleanup(true);
                List<Data> l2_data = new LinkedList<>();
                l2_data.add(mat6);
                ListObject l2 = new ListObject(l2_data);
                l1_data.add(l2);
            }

            ListObject l1 = new ListObject(l1_data);
            vars.put("l1", l1);

            varList.add("l1");

            // cleanup flag of inner matrix (m4)
            varStates.add(true);
            varStates.add(false);
            if (nestedList)
                varStates.add(true);
        }

        ExecutionContext ec = new ExecutionContext(vars);

        commonPinVariablesTest(ec, varList, varStates);
    }

    private void commonPinVariablesTest(ExecutionContext ec, List<String> varList, Queue<Boolean> varStatesExp) {
        Queue<Boolean> varStates = ec.pinVariables(varList);

        // check returned cleanupEnabled flags
        Assert.assertEquals(varStatesExp, varStates);

        // assert updated cleanupEnabled flag to false
        for (String varName : varList) {
            Data dat = ec.getVariable(varName);

            if (dat instanceof CacheableData<?>)
                Assert.assertFalse(((CacheableData<?>)dat).isCleanupEnabled());
            else if (dat instanceof ListObject) {
                assertListFlagsDisabled((ListObject)dat);
            }
        }

        ec.unpinVariables(varList, varStates);

        // check returned flags after unpinVariables()
        Queue<Boolean> varStates2 = ec.pinVariables(varList);
        Assert.assertEquals(varStatesExp, varStates2);
    }

    private void assertListFlagsDisabled(ListObject l) {
        for (Data dat : l.getData()) {
            if (dat instanceof CacheableData<?>)
                Assert.assertFalse(((CacheableData<?>)dat).isCleanupEnabled());
            else if (dat instanceof ListObject)
                assertListFlagsDisabled((ListObject)dat);
        }
    }
}
