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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCSingleZeros;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.junit.Test;

public class CustomColGroupTest {
	protected static final Log LOG = LogFactory.getLog(CustomColGroupTest.class.getName());

    @Test
    public void appendEmptyToSDCZero() {
        IColIndex i = ColIndexFactory.createI(3);
        AColGroup e = new ColGroupEmpty(i);
        AColGroup s = ColGroupSDCSingleZeros.create(i, 10, new PlaceHolderDict(1),
            OffsetFactory.createOffset(new int[] {5, 10}), null);

        AColGroup r = AColGroup.appendN(new AColGroup[] {e, s}, 20, 40);

        assertTrue(r instanceof ColGroupSDCSingleZeros);
        assertEquals(r.getColIndices(),i);
        assertEquals(((ColGroupSDCSingleZeros)r).getNumRows(), 40);

    }


    @Test
    public void appendEmptyToSDCZero2() {
        IColIndex i = ColIndexFactory.createI(3);
        AColGroup e = new ColGroupEmpty(i);
        AColGroup s = ColGroupSDCSingleZeros.create(i, 10, new PlaceHolderDict(1),
            OffsetFactory.createOffset(new int[] {5, 10}), null);

        AColGroup r = AColGroup.appendN(new AColGroup[] {e, s, e, e, s, s, e}, 20, 7*20);
        LOG.error(r);
        assertTrue(r instanceof ColGroupSDCSingleZeros);
        assertEquals(r.getColIndices(),i);
        assertEquals(((ColGroupSDCSingleZeros)r).getNumRows(), 7 * 20);

    }
}
