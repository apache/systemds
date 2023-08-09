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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.junit.Test;

public class CLALibSliceTest {
    protected static final Log LOG = LogFactory.getLog(CLALibSliceTest.class.getName());

    @Test
    public void sliceColumnsRanges() {
        List<AColGroup> gs = new ArrayList<AColGroup>();
        for(int i = 0; i < 10; i++) {
            gs.add(new ColGroupEmpty(ColIndexFactory.create(i * 10, i * 10 + 10)));
        }

        CompressedMatrixBlock cmb = new CompressedMatrixBlock(100, 100, -1, false, gs);

        CompressedMatrixBlock cmb2 = (CompressedMatrixBlock) cmb.slice(0, 99, 5, 49);
        assertEquals(49 - 4, cmb2.getNumColumns());
        assertEquals(5, cmb2.getColGroups().size());

        int countColumns = 0;
        for(AColGroup g : cmb2.getColGroups()) {
            IColIndex idx = g.getColIndices();
            countColumns += idx.size();
            assertTrue(idx.get(0) >= 0);
            assertTrue(idx.get(idx.size() - 1) < cmb.getNumColumns());
        }
        assertEquals(cmb2.getNumColumns(), countColumns);
    }

    @Test
    public void sliceSingleColumns() {
        List<AColGroup> gs = new ArrayList<AColGroup>();
        for(int i = 0; i < 50; i++) {
            gs.add(new ColGroupEmpty(ColIndexFactory.create(i, i + 1)));
        }

        CompressedMatrixBlock cmb = new CompressedMatrixBlock(100, 50, -1, false, gs);

        CompressedMatrixBlock cmb2 = (CompressedMatrixBlock) cmb.slice(0, 99, 5, 40);
        assertEquals(40 - 4, cmb2.getNumColumns());
        assertEquals(40 - 4, cmb2.getColGroups().size());

        int countColumns = 0;
        for(AColGroup g : cmb2.getColGroups()) {
            IColIndex idx = g.getColIndices();
            countColumns += idx.size();
            assertTrue(idx.get(0) >= 0);
            assertTrue(idx.get(idx.size() - 1) < cmb.getNumColumns());
        }
        assertEquals(cmb2.getNumColumns(), countColumns);
    }

    @Test
    public void sliceTwoColumns() {
        List<AColGroup> gs = new ArrayList<AColGroup>();
        for(int i = 0; i < 50;  i+=2) {
            gs.add(new ColGroupEmpty(ColIndexFactory.createI(i, i +1)));
        }

        CompressedMatrixBlock cmb = new CompressedMatrixBlock(100, 50, -1, false, gs);

        CompressedMatrixBlock cmb2 = (CompressedMatrixBlock) cmb.slice(0, 99, 5, 40);
        assertEquals(40 - 4, cmb2.getNumColumns());
        assertEquals((40 - 4) /2 + 1, cmb2.getColGroups().size());

        int countColumns = 0;
        for(AColGroup g : cmb2.getColGroups()) {
            IColIndex idx = g.getColIndices();
            countColumns += idx.size();
            assertTrue(idx.get(0) >= 0);
            assertTrue(idx.get(idx.size() - 1) < cmb.getNumColumns());
        }
        assertEquals(cmb2.getNumColumns(), countColumns);
    }


    @Test
    public void sliceTwoColumnsV2() {
        List<AColGroup> gs = new ArrayList<AColGroup>();
        gs.add(new ColGroupEmpty(ColIndexFactory.createI(0)));
        for(int i = 1; i < 51;  i+=2) {
            gs.add(new ColGroupEmpty(ColIndexFactory.createI(i, i +1)));
        }

        CompressedMatrixBlock cmb = new CompressedMatrixBlock(100, 51, -1, false, gs);

        CompressedMatrixBlock cmb2 = (CompressedMatrixBlock) cmb.slice(0, 99, 5, 40);
        assertEquals(40 - 4, cmb2.getNumColumns());
        assertEquals(18, cmb2.getColGroups().size());

        int countColumns = 0;
        for(AColGroup g : cmb2.getColGroups()) {
            IColIndex idx = g.getColIndices();
            countColumns += idx.size();
            assertTrue(idx.get(0) >= 0);
            assertTrue(idx.get(idx.size() - 1) < cmb.getNumColumns());
        }
        assertEquals(cmb2.getNumColumns(), countColumns);
    }
}
