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

package org.apache.sysds.test.component.compress.cost;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.junit.Test;

public class InstructionCounterTest {

	@Test
	public void testEmpty() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.toString();
		assertEquals(0, c.getScans());
	}

	@Test
	public void testScans() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incScans();
		c.toString();
		assertEquals(2, c.getScans());
	}

	@Test
	public void testScans_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans(3);
		c.toString();
		assertEquals(3, c.getScans());
	}

	@Test
	public void testDecompressions() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
	}

	@Test
	public void testDecompressions_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions(4);
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(4, c.getDecompressions());
	}

	@Test
	public void testOverlappingDecompressions() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
	}

	@Test
	public void testOverlappingDecompressions_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions(42);
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(42, c.getOverlappingDecompressions());
	}

	@Test
	public void testLeftMultiplications() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM();
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(1, c.getLeftMultiplications());

	}

	@Test
	public void testLeftMultiplications_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
	}

	@Test
	public void testRightMultiplications() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM();
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(1, c.getRightMultiplications());
	}

	@Test
	public void testRightMultiplications_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
	}

	@Test
	public void testCMM() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.incCMM();
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
		assertEquals(1, c.getCompressedMultiplications());
	}

	@Test
	public void testCMM_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.incCMM(42);
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
		assertEquals(42, c.getCompressedMultiplications());
	}

	@Test
	public void testDictOps() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.incCMM();
		c.incDictOps();
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
		assertEquals(1, c.getCompressedMultiplications());
		assertEquals(1, c.getDictionaryOps());
	}

	@Test
	public void testDictOps_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.incCMM();
		c.incDictOps(222);
		c.toString();
		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
		assertEquals(1, c.getCompressedMultiplications());
		assertEquals(222, c.getDictionaryOps());
	}

	@Test
	public void testIndexing() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.incCMM();
		c.incDictOps();
		c.incIndexOp();
		c.toString();

		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
		assertEquals(1, c.getCompressedMultiplications());
		assertEquals(1, c.getDictionaryOps());
		assertEquals(1, c.getIndexing());
	}

	@Test
	public void testIndexing_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incLMM(30);
		c.incRMM(23);
		c.incCMM();
		c.incDictOps();
		c.incIndexOp(425);
		c.toString();

		assertEquals(1, c.getScans());
		assertEquals(1, c.getDecompressions());
		assertEquals(1, c.getOverlappingDecompressions());
		assertEquals(30, c.getLeftMultiplications());
		assertEquals(23, c.getRightMultiplications());
		assertEquals(1, c.getCompressedMultiplications());
		assertEquals(1, c.getDictionaryOps());
		assertEquals(425, c.getIndexing());
	}

	@Test
	public void testDensifying() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.toString();

		assertEquals(1, c.getScans());
		assertFalse(c.isDensifying());
	}

	@Test
	public void testDensifying_2() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.setDensifying(true);
		c.toString();

		assertEquals(1, c.getScans());
		assertTrue(c.isDensifying());
	}

	@Test
	public void testVarious() {
		InstructionTypeCounter c = new InstructionTypeCounter();
		c.incScans();
		c.incOverlappingDecompressions();
		c.incOverlappingDecompressions();
		c.incDecompressions();
		c.incDecompressions();
		c.incLMM(30);
		c.incDecompressions();
		c.incOverlappingDecompressions();
		c.incRMM(23);
		c.incCMM();
		c.incDictOps();
		c.incRMM(22);
		c.incDictOps();
		c.incScans();
		c.incLMM(32);
		c.incCMM();
		c.incOverlappingDecompressions();
		c.incDictOps();
		c.incIndexOp();

		c.toString();

		assertEquals(2, c.getScans());
		assertEquals(3, c.getDecompressions());
		assertEquals(4, c.getOverlappingDecompressions());
		assertEquals(62, c.getLeftMultiplications());
		assertEquals(45, c.getRightMultiplications());
		assertEquals(2, c.getCompressedMultiplications());
		assertEquals(3, c.getDictionaryOps());
		assertEquals(1, c.getIndexing());
	}

	@Test
	public void testDicrectConstructor() {

		InstructionTypeCounter c = new InstructionTypeCounter(2, 3, 4, 62, 45, 2, 3, 1, false);

		assertEquals(2, c.getScans());
		assertEquals(3, c.getDecompressions());
		assertEquals(4, c.getOverlappingDecompressions());
		assertEquals(62, c.getLeftMultiplications());
		assertEquals(45, c.getRightMultiplications());
		assertEquals(2, c.getCompressedMultiplications());
		assertEquals(3, c.getDictionaryOps());
		assertEquals(1, c.getIndexing());
		assertTrue(!c.isDensifying());
	}

	@Test
	public void testDicrectConstructor_2() {
		InstructionTypeCounter c = new InstructionTypeCounter(2, 3, 4, 62, 45, 2, 3, 1, true);
		assertEquals(2, c.getScans());
		assertEquals(3, c.getDecompressions());
		assertEquals(4, c.getOverlappingDecompressions());
		assertEquals(62, c.getLeftMultiplications());
		assertEquals(45, c.getRightMultiplications());
		assertEquals(2, c.getCompressedMultiplications());
		assertEquals(3, c.getDictionaryOps());
		assertEquals(1, c.getIndexing());
		assertTrue(c.isDensifying());
	}
}
