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
package org.apache.sysds.test.component.compress.io;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.lib.CLALibSeparator;
import org.apache.sysds.runtime.compress.lib.CLALibSeparator.SeparatedGroups;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class SeparateDictionariesAndIndexes {

	protected static final Log LOG = LogFactory.getLog(IOSpark.class.getName());

	@Test
	public void separate() {
		try {

			MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 5, 0, 9, 1.0, 1342);
			mb = TestUtils.ceil(mb);
			mb = mb.append(mb).append(mb);
			CompressedMatrixBlock cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb).getLeft();
			List<AColGroup> gs = cmb.getColGroups();

			SeparatedGroups s = CLALibSeparator.split(cmb.getColGroups());
			assertTrue(s.dicts.size() > 0);
			assertTrue(size(gs) >= size(s.indexStructures));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	private long size(List<AColGroup> g) {
		long s = 0;
		for(AColGroup gs : g)
			s += gs.estimateInMemorySize();
		return s;
	}

}
