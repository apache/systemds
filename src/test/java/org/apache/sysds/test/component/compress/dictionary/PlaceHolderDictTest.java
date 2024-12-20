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

package org.apache.sysds.test.component.compress.dictionary;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;
import org.junit.Test;

public class PlaceHolderDictTest {

	PlaceHolderDict d = new PlaceHolderDict(1);

	@Test(expected = Exception.class)
	public void getValues() {
		d.getValues();
	}

	@Test(expected = Exception.class)
	public void getValue() {
		d.getValue(1);
	}

	@Test(expected = Exception.class)
	public void getValue2() {
		d.getValue(1, 2, 3);
	}

	@Test
	public void getInMemorySize() {
		assertEquals(16 + 4, d.getInMemorySize());
	}

	@Test(expected = Exception.class)
	public void aggregate() {
		d.aggregate(1, null);
	}

	@Test(expected = Exception.class)
	public void aggregateWithReference() {
		d.aggregateWithReference(1, null, null, true);
	}

	@Test(expected = Exception.class)
	public void aggregateRows() {
		d.aggregateRows(null, 1);
	}

	@Test(expected = Exception.class)
	public void aggregateRowsWithDefault() {
		d.aggregateRowsWithDefault(null, null);
	}

	@Test(expected = Exception.class)
	public void aggregateRowsWithReference() {
		d.aggregateRowsWithReference(null, null);
	}

	@Test(expected = Exception.class)
	public void aggregateCols() {
		d.aggregateCols(null, null, null);
	}

	@Test(expected = Exception.class)
	public void aggregateColsWithReference() {
		d.aggregateColsWithReference(null, null, null, null, false);
	}

	@Test(expected = Exception.class)
	public void applyScalarOp() {
		d.applyScalarOp(null);
	}

	@Test(expected = Exception.class)
	public void applyScalarOpAndAppend() {
		d.applyScalarOpAndAppend(null, 1, 1);
	}

	@Test(expected = Exception.class)
	public void applyUnaryOp() {
		d.applyUnaryOp(null);
	}

	@Test(expected = Exception.class)
	public void applyUnaryOpAndAppend() {
		d.applyUnaryOpAndAppend(null, 1, 1);
	}

	@Test(expected = Exception.class)
	public void applyScalarOpWithReference() {
		d.applyScalarOpWithReference(null, null, null);
	}

	@Test(expected = Exception.class)
	public void applyUnaryOpWithReference() {
		d.applyUnaryOpWithReference(null, null, null);
	}

	@Test(expected = Exception.class)
	public void binOpLeft() {
		d.binOpLeft(null, null, null);
	}

	@Test(expected = Exception.class)
	public void binOpLeftAndAppend() {
		d.binOpLeftAndAppend(null, null, null);
	}

	@Test(expected = Exception.class)
	public void binOpLeftWithReference() {
		d.binOpLeftWithReference(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void binOpRight() {
		d.binOpRight(null, null, null);
	}

	@Test(expected = Exception.class)
	public void binOpRightAndAppend() {
		d.binOpRightAndAppend(null, null, null);
	}

	@Test(expected = Exception.class)
	public void binOpRight2() {
		d.binOpRight(null, null);
	}

	@Test(expected = Exception.class)
	public void binOpRightWithReference() {
		d.binOpRightWithReference(null, null, null, null, null);
	}

	@Test
	public void getExactSizeOnDisk() {
		assertEquals(5, d.getExactSizeOnDisk());
	}

	@Test(expected = Exception.class)
	public void getDictType() {
		d.getDictType();
	}

	@Test
	public void getNumberOfValues() {
		assertEquals(1, d.getNumberOfValues(1));
	}

	@Test(expected = Exception.class)
	public void sumAllRowsToDouble() {
		d.sumAllRowsToDouble(3);
	}

	@Test(expected = Exception.class)
	public void sumAllRowsToDoubleWithDefault() {
		d.sumAllRowsToDoubleWithDefault(null);
	}

	@Test(expected = Exception.class)
	public void sumAllRowsToDoubleWithReference() {
		d.sumAllRowsToDoubleWithReference(null);
	}

	@Test(expected = Exception.class)
	public void sumAllRowsToDoubleSq() {
		d.sumAllRowsToDoubleSq(1);
	}

	@Test(expected = Exception.class)
	public void sumAllRowsToDoubleSqWithDefault() {
		d.sumAllRowsToDoubleSqWithDefault(null);
	}

	@Test(expected = Exception.class)
	public void sumAllRowsToDoubleSqWithReference() {
		d.sumAllRowsToDoubleSqWithReference(null);
	}

	@Test(expected = Exception.class)
	public void productAllRowsToDouble() {
		d.productAllRowsToDouble(22);
	}

	@Test(expected = Exception.class)
	public void productAllRowsToDoubleWithDefault() {
		d.productAllRowsToDoubleWithDefault(null);
	}

	@Test(expected = Exception.class)
	public void productAllRowsToDoubleWithReference() {
		d.productAllRowsToDoubleWithReference(null);
	}

	@Test(expected = Exception.class)
	public void colSum() {
		d.colSum(null, null, null);
	}

	@Test(expected = Exception.class)
	public void colSumSq() {
		d.colSumSq(null, null, null);
	}

	@Test(expected = Exception.class)
	public void colSumSqWithReference() {
		d.colSumSqWithReference(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void sum() {
		d.sum(null, 1);
	}

	@Test(expected = Exception.class)
	public void sumSq() {
		d.sumSq(null, 1);
	}

	@Test(expected = Exception.class)
	public void sumSqWithReference() {
		d.sumSqWithReference(null, null);
	}

	@Test
	public void getString() {
		assertEquals("", d.getString(1));
	}

	@Test(expected = Exception.class)
	public void sliceOutColumnRange() {
		d.sliceOutColumnRange(1, 1, 1);
	}

	@Test(expected = Exception.class)
	public void containsValue() {
		d.containsValue(1);
	}

	@Test(expected = Exception.class)
	public void containsValueWithReference() {
		d.containsValueWithReference(1, null);
	}

	@Test
	public void getNumberNonZeros() {
		assertEquals(-1, d.getNumberNonZeros(null, 1));
	}

	@Test(expected = Exception.class)
	public void getNumberNonZerosWithReference() {
		d.getNumberNonZerosWithReference(null, null, 1);
	}

	@Test(expected = Exception.class)
	public void addToEntry() {
		d.addToEntry(null, 1, 1, 1);
	}

	@Test(expected = Exception.class)
	public void addToEntry2() {
		d.addToEntry(null, 1, 1, 1, 2);
	}

	@Test(expected = Exception.class)
	public void addToEntryVectorized() {
		d.addToEntryVectorized(null, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1);
	}

	@Test(expected = Exception.class)
	public void subtractTuple() {
		d.subtractTuple(null);
	}

	@Test(expected = Exception.class)
	public void getMBDict() {
		d.getMBDict(1);
	}

	@Test(expected = Exception.class)
	public void scaleTuples() {
		d.scaleTuples(null, 1);
	}

	@Test(expected = Exception.class)
	public void preaggValuesFromDense() {
		d.preaggValuesFromDense(1, null, null, null, 1);
	}

	@Test(expected = Exception.class)
	public void replace() {
		d.replace(1, 1, 1);
	}

	@Test(expected = Exception.class)
	public void replaceWithReference() {
		d.replaceWithReference(1, 1, null);
	}

	@Test(expected = Exception.class)
	public void product() {
		d.product(null, null, 1);
	}

	@Test(expected = Exception.class)
	public void productWithDefault() {
		d.productWithDefault(null, null, null, 1);
	}

	@Test(expected = Exception.class)
	public void productWithReference() {
		d.productWithReference(null, null, null, 1);
	}

	@Test(expected = Exception.class)
	public void colProduct() {
		d.colProduct(null, null, null);
	}

	@Test(expected = Exception.class)
	public void colProductWithReference() {
		d.colProductWithReference(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void centralMoment() {
		d.centralMoment(null, null, 1);
	}

	@Test(expected = Exception.class)
	public void centralMoment2() {
		d.centralMoment(null, null, null, 1);
	}

	@Test(expected = Exception.class)
	public void centralMomentWithDefault() {
		d.centralMomentWithDefault(null, null, 1, 1);
	}

	@Test(expected = Exception.class)
	public void centralMomentWithDefault2() {
		d.centralMomentWithDefault(null, null, null, 1, 1);
	}

	@Test(expected = Exception.class)
	public void centralMomentWithReference() {
		d.centralMomentWithReference(null, null, 1, 1);
	}

	@Test(expected = Exception.class)
	public void centralMomentWithReference2() {
		d.centralMomentWithReference(null, null, null, 1, 1);
	}

	@Test(expected = Exception.class)
	public void rexpandCols() {
		d.rexpandCols(1, false, false, 1);
	}

	@Test(expected = Exception.class)
	public void rexpandColsWithReference() {
		d.rexpandColsWithReference(1, false, false, 1);
	}

	@Test(expected = Exception.class)
	public void getSparsity() {
		d.getSparsity();
	}

	@Test(expected = Exception.class)
	public void multiplyScalar() {
		d.multiplyScalar(1, null, 1, 1, null);
	}

	@Test(expected = Exception.class)
	public void TSMMWithScaling() {
		d.TSMMWithScaling(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void MMDict() {
		d.MMDict(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void MMDictDense() {
		d.MMDictDense(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void MMDictSparse() {
		d.MMDictSparse(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void TSMMToUpperTriangle() {
		d.TSMMToUpperTriangle(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void TSMMToUpperTriangleDense() {
		d.TSMMToUpperTriangleDense(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void TSMMToUpperTriangleSparse() {
		d.TSMMToUpperTriangleSparse(null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void TSMMToUpperTriangleScaling() {
		d.TSMMToUpperTriangleScaling(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void TSMMToUpperTriangleDenseScaling() {
		d.TSMMToUpperTriangleDenseScaling(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void TSMMToUpperTriangleSparseScaling() {
		d.TSMMToUpperTriangleSparseScaling(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void cbind() {
		d.cbind(null, 1);
	}

	@Test
	public void equals() {
		assertTrue(!d.equals(Dictionary.create(new double[2])));
		assertTrue(d.equals(new PlaceHolderDict(2)));
	}

	@Test
	public void equals2() {
		assertTrue(!d.equals(new double[3]));
	}

	@Test(expected = Exception.class)
	public void reorder() {
		d.reorder(null);
	}

	@Test
	public void cloneTest() {
		assertTrue(d.equals(d.clone()));
	}

	@Test(expected = Exception.class)
	public void MMDictScaling() {
		d.MMDictScaling(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void MMDictScalingDense() {
		d.MMDictScalingDense(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void MMDictScalingSparse() {
		d.MMDictScalingSparse(null, null, null, null, null);
	}

	@Test(expected = Exception.class)
	public void putDense() {
		d.putDense(null, 1, 1, 1, null);
	}

	@Test(expected = Exception.class)
	public void putSparse() {
		d.putSparse(null, 1, 1, 1, null);
	}

	@Test
	public void testSerialization() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			d.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			IDictionary d2 = DictionaryFactory.read(fis);
			assertTrue(d.equals(d2));
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}
}
