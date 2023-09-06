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

package org.apache.sysds.test.component.compress.util;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.utils.ACount.DCounts;
import org.junit.Test;

public class hashTest {

    @Test
    public void t1() {
        int a = DCounts.hashIndex(Double.NaN);
        assertTrue(a >= 0);
    }

    @Test
    public void t2() {
        int a = DCounts.hashIndex(Double.POSITIVE_INFINITY);
        assertTrue(a >= 0);
    }

    @Test
    public void t3() {
        int a = DCounts.hashIndex(Double.NEGATIVE_INFINITY);
        assertTrue(a >= 0);
    }

    @Test
    public void t4() {
        int a = DCounts.hashIndex(Double.MIN_NORMAL);
        assertTrue(a >= 0);
    }

    @Test
    public void t5() {
        int a = DCounts.hashIndex(-Double.MIN_NORMAL);
        assertTrue(a >= 0);
    }

    @Test
    public void t6() {
        int a = DCounts.hashIndex(-1);
        assertTrue(a >= 0);
    }

    @Test
    public void t7() {
        int a = DCounts.hashIndex(-0.000000000000000000000000000001);
        assertTrue(a >= 0);
    }

    @Test
    public void t8() {
        int a = DCounts.hashIndex(-0.0);
        // this iss.... annoying see.
        // https://stackoverflow.com/questions/18565485/why-is-absolute-of-integer-min-value-equivalent-to-integer-min-value
        assertFalse(a + " should be greater or equal to zero", a >= 0);
    }

}
