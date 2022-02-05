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

package org.apache.sysds.test.functions.countDistinct;

import org.apache.sysds.common.Types;

public class CountDistinctApproxRow extends CountDistinctRowOrColBase {

    private final static String TEST_NAME = "countDistinctApproxRow";
    private final static String TEST_DIR = "functions/countDistinct/";
    private final static String TEST_CLASS_DIR = TEST_DIR + CountDistinctApproxRow.class.getSimpleName() + "/";

    @Override
    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    @Override
    protected String getTestName() {
        return TEST_NAME;
    }

    @Override
    protected String getTestDir() {
        return TEST_DIR;
    }

    @Override
    protected Types.Direction getDirection() {
        return Types.Direction.Row;
    }

    @Override
    public void setUp() {
        super.addTestConfiguration();
    }
}
