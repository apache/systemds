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

package org.apache.sysds.test.component.compress.colgroup.scheme;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.ConstScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.EmptyScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class NegativeSchemeTest {
	protected final Log LOG = LogFactory.getLog(NegativeSchemeTest.class.getName());

	final ICLAScheme sh;

	public NegativeSchemeTest() {
		sh = new EmptyScheme(ColIndexFactory.createI(3));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidColumnApply() {
		sh.encode(null, ColIndexFactory.create(new int[] {1, 2}));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidColumnApply_2() {
		sh.encode(null, ColIndexFactory.create(new int[] {1, 2, 5, 5}));
	}

	@Test(expected = NullPointerException.class)
	public void testNull() {
		sh.encode(null, null);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidColumnApply_NumberColumns() {
		sh.encode(new MatrixBlock(0, 0, false), ColIndexFactory.create(new int[] {1}));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidColumnApplyT() {
		sh.encodeT(null, ColIndexFactory.create(new int[] {1, 2}));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidColumnApply_2T() {
		sh.encodeT(null, ColIndexFactory.create(new int[] {1, 2, 5, 5}));
	}

	@Test(expected = NullPointerException.class)
	public void testNullT() {
		sh.encodeT(null, null);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidColumnApplyT_NumberColumns() {
		sh.encodeT(new MatrixBlock(0, 0, false), ColIndexFactory.create(new int[] {1}));
	}

	@Test(expected = RuntimeException.class)
	public void testConstCreate() {
		ConstScheme.create(null, null);
	}
}
