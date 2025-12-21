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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.EmptyEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.SparseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.ConstEncoding;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class EncodeDeltaTest {

	@Test
	public void testCreateFromMatrixBlockDeltaBasic() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("First row [10,20] stored as-is, deltas [1,1] for rows 1-2, so 2 unique: [10,20] and [1,1]", 2, encoding.getUnique());
		assertTrue("Encoding should be dense", encoding.isDense());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaWithSampleSize() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 5; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 3);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("Sample size is 3, so should process 3 rows", 3, ((DenseEncoding) encoding).getMap().size());
		assertTrue("Should have at least 1 unique delta value", encoding.getUnique() >= 1);
		assertTrue("Should have at most 3 unique delta values (one per row)", encoding.getUnique() <= 3);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaFirstRowAsIs() {
		MatrixBlock mb = new MatrixBlock(2, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 5);
		mb.set(0, 1, 10);
		mb.set(1, 0, 5);
		mb.set(1, 1, 10);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("First row [5,10] stored as-is, delta [0,0] for row 1. Map has 2 unique: [5,10] and [0,0]. With zero=true, unique = 2 + 1 = 3", 3, encoding.getUnique());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaConstantDeltas() {
		MatrixBlock mb = new MatrixBlock(4, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);
		mb.set(3, 0, 13);
		mb.set(3, 1, 23);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("First row [10,20] stored as-is, all deltas are [1,1], so 2 unique: [10,20] and [1,1]", 2, encoding.getUnique());
		assertTrue("Encoding should be dense", encoding.isDense());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSingleRow() {
		MatrixBlock mb = new MatrixBlock(1, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		// Single row results in ConstEncoding because there is only 1 unique value (the row itself)
		assertTrue("Single row should result in ConstEncoding", encoding instanceof ConstEncoding);
		assertEquals("Single row has no deltas, so should have 1 unique value (the row itself)", 1, encoding.getUnique());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSparse() {
		MatrixBlock mb = new MatrixBlock(3, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(2, 1, 22);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Sparse input may result in SparseEncoding or DenseEncoding", 
			encoding instanceof DenseEncoding || encoding instanceof SparseEncoding);
		assertTrue("Should have at least 1 unique value", encoding.getUnique() >= 1);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaColumnSelection() {
		MatrixBlock mb = new MatrixBlock(3, 4, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(0, 2, 30);
		mb.set(0, 3, 40);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(1, 2, 31);
		mb.set(1, 3, 41);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);
		mb.set(2, 2, 32);
		mb.set(2, 3, 42);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(0, 2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("Selected columns 0 and 2: first row [10,30] stored as-is, deltas [1,1] for rows 1-2, so 2 unique: [10,30] and [1,1]", 2, encoding.getUnique());
		assertEquals("Should have 3 rows in mapping", 3, ((DenseEncoding) encoding).getMap().size());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaNegativeValues() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 8);
		mb.set(1, 1, 15);
		mb.set(2, 0, 12);
		mb.set(2, 1, 25);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		// Deltas: R0=[10,20], R1=[-2,-5], R2=[4,10] -> 3 unique values
		assertEquals("Should have 3 unique values", 3, encoding.getUnique());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaZeros() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 5);
		mb.set(0, 1, 0);
		mb.set(1, 0, 5);
		mb.set(1, 1, 0);
		mb.set(2, 0, 0);
		mb.set(2, 1, 5);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding or SparseEncoding", 
			encoding instanceof DenseEncoding || encoding instanceof SparseEncoding);
		assertTrue("Should have at least 1 unique value", encoding.getUnique() >= 1);
	}


	@Test
	public void testCreateFromMatrixBlockDeltaLargeMatrix() {
		MatrixBlock mb = new MatrixBlock(100, 3, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 100; i++) {
			mb.set(i, 0, i);
			mb.set(i, 1, i * 2);
			mb.set(i, 2, i * 3);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(3));
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("First row [0,0,0] stored as-is, all deltas are [1,2,3]. Map has 2 unique: [0,0,0] and [1,2,3]. All rows have non-zero deltas, so offsets.size()=100=ru, zero=false, unique=2", 2, encoding.getUnique());
		assertEquals("Should have 100 rows in mapping", 100, ((DenseEncoding) encoding).getMap().size());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSampleSizeSmaller() {
		MatrixBlock mb = new MatrixBlock(10, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 10; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 5);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("Sample size is 5, so should process 5 rows", 5, ((DenseEncoding) encoding).getMap().size());
		assertEquals("First row [10,20] stored as-is, all deltas are [1,1], so 2 unique: [10,20] and [1,1]", 2, encoding.getUnique());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSampleSizeLarger() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 5; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 10);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Encoding should be DenseEncoding", encoding instanceof DenseEncoding);
		assertEquals("Sample size 10 > matrix rows 5, so should process all 5 rows", 5, ((DenseEncoding) encoding).getMap().size());
		assertEquals("First row [10,20] stored as-is, all deltas are [1,1], so 2 unique: [10,20] and [1,1]", 2, encoding.getUnique());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaEmptyMatrix() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		// Empty matrix (all zeros) is constant 0 in delta encoding
		assertTrue("Empty matrix should result in ConstEncoding or EmptyEncoding", 
			encoding instanceof ConstEncoding || encoding instanceof EmptyEncoding);
		// Both ConstEncoding(0) and EmptyEncoding return 1 unique value (the zero tuple)
		assertEquals("Encoding of zeros should have 1 unique value", 1, encoding.getUnique());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaEmptyMatrixSparse() {
		MatrixBlock mb = new MatrixBlock(5, 2, true);
		mb.setNonZeros(0);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull("Encoding should not be null", encoding);
		// Empty sparse matrix is also constant 0
		assertTrue("Empty sparse matrix should result in ConstEncoding or EmptyEncoding", 
			encoding instanceof ConstEncoding || encoding instanceof EmptyEncoding);
		// Both ConstEncoding(0) and EmptyEncoding return 1 unique value (the zero tuple)
		assertEquals("Encoding of zeros should have 1 unique value", 1, encoding.getUnique());
	}

	@Test
	public void testCombineTwoDenseDeltaEncodings() {
		MatrixBlock mb1 = new MatrixBlock(3, 1, false);
		mb1.allocateDenseBlock();
		mb1.set(0, 0, 10);
		mb1.set(1, 0, 11);
		mb1.set(2, 0, 12);

		MatrixBlock mb2 = new MatrixBlock(3, 1, false);
		mb2.allocateDenseBlock();
		mb2.set(0, 0, 20);
		mb2.set(1, 0, 21);
		mb2.set(2, 0, 22);

		IEncode enc1 = EncodingFactory.createFromMatrixBlockDelta(mb1, false, ColIndexFactory.create(1));
		IEncode enc2 = EncodingFactory.createFromMatrixBlockDelta(mb2, false, ColIndexFactory.create(1));

		assertNotNull("First encoding should not be null", enc1);
		assertNotNull("Second encoding should not be null", enc2);
		assertTrue("First encoding should be DenseEncoding", enc1 instanceof DenseEncoding);
		assertTrue("Second encoding should be DenseEncoding", enc2 instanceof DenseEncoding);

		IEncode combined = enc1.combine(enc2);
		assertNotNull("Combined encoding should not be null", combined);
		assertTrue("Combined encoding should be DenseEncoding", combined instanceof DenseEncoding);
		assertTrue("Combined unique count should be at least max of inputs", 
			combined.getUnique() >= Math.max(enc1.getUnique(), enc2.getUnique()));
		assertTrue("Combined unique count should be at most product of inputs", 
			combined.getUnique() <= enc1.getUnique() * enc2.getUnique());
		assertEquals("Combined mapping should have same size as input", 
			((DenseEncoding) enc1).getMap().size(), ((DenseEncoding) combined).getMap().size());
	}

	@Test
	public void testCombineDenseDeltaEncodingWithEmpty() {
		MatrixBlock mb = new MatrixBlock(3, 1, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(1, 0, 11);
		mb.set(2, 0, 12);

		IEncode enc1 = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(1));
		IEncode enc2 = new EmptyEncoding();

		assertNotNull("First encoding should not be null", enc1);
		assertTrue("First encoding should be DenseEncoding", enc1 instanceof DenseEncoding);

		IEncode combined = enc1.combine(enc2);
		assertNotNull("Combined encoding should not be null", combined);
		assertEquals("Combining with EmptyEncoding should return original encoding", enc1, combined);
	}

	@Test
	public void testCombineDenseDeltaEncodingWithConst() {
		MatrixBlock mb = new MatrixBlock(3, 1, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(1, 0, 11);
		mb.set(2, 0, 12);

		IEncode enc1 = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(1));
		
		MatrixBlock constMb = new MatrixBlock(3, 1, false);
		constMb.allocateDenseBlock();
		constMb.set(0, 0, 5);
		constMb.set(1, 0, 5);
		constMb.set(2, 0, 5);
		IEncode enc2 = EncodingFactory.createFromMatrixBlock(constMb, false, ColIndexFactory.create(1));

		assertNotNull("First encoding should not be null", enc1);
		assertTrue("First encoding should be DenseEncoding", enc1 instanceof DenseEncoding);
		assertTrue("Second encoding should be ConstEncoding", enc2 instanceof ConstEncoding);

		IEncode combined = enc1.combine(enc2);
		assertNotNull("Combined encoding should not be null", combined);
		assertEquals("Combining with ConstEncoding should return original encoding", enc1, combined);
	}

	@Test
	public void testCombineDenseDeltaEncodingsWithDifferentDeltas() {
		MatrixBlock mb1 = new MatrixBlock(4, 1, false);
		mb1.allocateDenseBlock();
		mb1.set(0, 0, 1);
		mb1.set(1, 0, 2);
		mb1.set(2, 0, 4);
		mb1.set(3, 0, 8);

		MatrixBlock mb2 = new MatrixBlock(4, 1, false);
		mb2.allocateDenseBlock();
		mb2.set(0, 0, 10);
		mb2.set(1, 0, 20);
		mb2.set(2, 0, 40);
		mb2.set(3, 0, 80);

		IEncode enc1 = EncodingFactory.createFromMatrixBlockDelta(mb1, false, ColIndexFactory.create(1));
		IEncode enc2 = EncodingFactory.createFromMatrixBlockDelta(mb2, false, ColIndexFactory.create(1));

		assertNotNull("First encoding should not be null", enc1);
		assertNotNull("Second encoding should not be null", enc2);
		assertTrue("First encoding should be DenseEncoding", enc1 instanceof DenseEncoding);
		assertTrue("Second encoding should be DenseEncoding", enc2 instanceof DenseEncoding);

		IEncode combined = enc1.combine(enc2);
		assertNotNull("Combined encoding should not be null", combined);
		assertTrue("Combined encoding should be DenseEncoding", combined instanceof DenseEncoding);
		assertEquals("Combined mapping should have same size as input", 
			4, ((DenseEncoding) combined).getMap().size());
	}

	@Test
	public void testCreateFromMatrixBlockDeltaDensePath() {
		MatrixBlock mb = new MatrixBlock(10, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);
		mb.set(3, 0, 13);
		mb.set(3, 1, 23);
		mb.set(4, 0, 14);
		mb.set(4, 1, 24);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 10);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Should result in DenseEncoding (5 non-zero rows >= 10/4=2.5, so dense path)", 
			encoding instanceof DenseEncoding);
		assertTrue("Should have at least 1 unique value", encoding.getUnique() >= 1);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaEmptyEncoding() {
		MatrixBlock mb = new MatrixBlock(10, 2, true);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 10);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Empty matrix should result in EmptyEncoding", encoding instanceof EmptyEncoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaConstEncoding() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 5; i++) {
			mb.set(i, 0, 10);
			mb.set(i, 1, 20);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 5);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Constant matrix with delta encoding: first row is absolute [10,20], rest are deltas [0,0], so map.size()=2, not ConstEncoding", 
			encoding instanceof DenseEncoding || encoding instanceof SparseEncoding);
		assertTrue("Should have 2 unique values (first row absolute, rest are zero deltas)", encoding.getUnique() >= 2);
	}


	@Test
	public void testCreateFromMatrixBlockDeltaSparseEncoding() {
		MatrixBlock mb = new MatrixBlock(20, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 20);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Sparse matrix with few non-zero rows (3 < 20/4=5) should result in SparseEncoding", 
			encoding instanceof SparseEncoding);
		assertTrue("Should have at least 1 unique value", encoding.getUnique() >= 1);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaDenseWithZero() {
		MatrixBlock mb = new MatrixBlock(10, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);
		mb.set(3, 0, 13);
		mb.set(3, 1, 23);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 10);
		assertNotNull("Encoding should not be null", encoding);
		assertTrue("Sparse matrix with some non-zero rows (4 >= 10/4=2.5 but 4 < 10) should result in DenseEncoding with zero=true", 
			encoding instanceof DenseEncoding);
		assertTrue("Should have at least 1 unique value", encoding.getUnique() >= 1);
	}

}

