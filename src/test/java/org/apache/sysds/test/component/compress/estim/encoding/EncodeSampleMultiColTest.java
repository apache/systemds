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

package org.apache.sysds.test.component.compress.estim.encoding;

import static org.junit.Assert.fail;

import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public abstract class EncodeSampleMultiColTest extends EncodeSampleTest {

	public IEncode fh;
	public IEncode sh;

	public EncodeSampleMultiColTest(MatrixBlock m, boolean t, int u, IEncode e, IEncode fh, IEncode sh) {
		super(m, t, u, e);
		this.fh = fh;
		this.sh = sh;
	}

	@Test
	public void testPartJoinEqualToFullRead() {
		try {

			partJoinVerification(fh.combine(sh));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testPartJoinEqualToFullReadLeft() {
		try {

			partJoinVerification(sh.combine(fh));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinWithFirstSubpart() {
		try {

			// again a test that does not make sense since joining with subpart results in equivalent but it is a valid
			// test
			partJoinVerification(e.combine(fh));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinWithSecondSubpart() {
		try {

			// joining with subpart results in equivalent but it is a valid test
			partJoinVerification(e.combine(sh));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinWithFirstSubpartLeft() {
		try {

			// joining with subpart results in equivalent but it is a valid test
			partJoinVerification(fh.combine(e));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinWithSecondSubpartLeft() {
		try {
			// joining with subpart results in equivalent but it is a valid test
			partJoinVerification(sh.combine(e));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void partJoinVerification(IEncode er) {
		boolean incorrectUnique = e.getUnique() != er.getUnique();

		er.extractFacts(10000, 1.0, 1.0, new CompressionSettingsBuilder().create());

		if(incorrectUnique) {
			StringBuilder sb = new StringBuilder();
			sb.append("\nFailed joining sub parts to recreate whole.");
			sb.append("\nexpected unique:" + e.getUnique() + " got:" + er.getUnique());

			sb.append("\n\nRead:");
			sb.append(e);
			sb.append("\nJoined:");
			sb.append(er);
			sb.append("\n");
			sb.append(m);
			sb.append("\n\nsubParts:\n");
			sb.append(sh);
			sb.append("\n");
			sb.append(fh);
			fail(sb.toString());
		}
	}
}
