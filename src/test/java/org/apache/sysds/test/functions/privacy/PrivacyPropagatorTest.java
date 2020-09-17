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

package org.apache.sysds.test.functions.privacy;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockLFP64;
import org.apache.sysds.runtime.data.DenseBlockLInt64;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;
import org.apache.sysds.runtime.privacy.PrivacyPropagator.OperatorType;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

import static org.junit.Assert.*;

public class PrivacyPropagatorTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		// TODO Auto-generated method stub

	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained2(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained3(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained4(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained5(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3,8},new long[]{2,5}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral2(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3,8},new long[]{2,5}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained(){
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertTrue("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
		assertTrue("Merged constraint should not contain privacy level PrivateAggregation", mergedConstraint.getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation).length == 0);
		Map<DataRange, PrivacyLevel> outputElement1 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,0});
		Map<DataRange, PrivacyLevel> outputElement2 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,1});
		Map<DataRange, PrivacyLevel> outputElement3 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,2});
		assertEquals(1, outputElement1.size());
		assertEquals(1, outputElement2.size());
		assertEquals(1, outputElement3.size());
		assertTrue("Privacy level of element 1 is Private", outputElement1.containsValue(PrivacyLevel.Private));
		assertTrue("Privacy level of element 2 is Private", outputElement2.containsValue(PrivacyLevel.Private));
		assertTrue("Privacy level of element 3 is Private", outputElement3.containsValue(PrivacyLevel.Private));
		Map<DataRange, PrivacyLevel> expectedEmpty = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{2,0}, new long[]{3,2}));
		assertTrue("Any other index has no privacy constraint", expectedEmpty.isEmpty() ||
			(!expectedEmpty.containsValue(PrivacyLevel.Private)
				&& !expectedEmpty.containsValue(PrivacyLevel.PrivateAggregation)));
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained2(){
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertTrue("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
		assertTrue("Merged constraint should not contain privacy level PrivateAggregation", mergedConstraint.getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation).length == 0);
		Map<DataRange, PrivacyLevel> outputRange = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{0,0},new long[]{3,1}));
		assertTrue("Privacy level is Private", outputRange.containsValue(PrivacyLevel.Private));
		Map<DataRange, PrivacyLevel> expectedEmpty = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{0,2}, new long[]{3,2}));
		assertTrue("Any other index has no privacy constraint", expectedEmpty.isEmpty() ||
			(!expectedEmpty.containsValue(PrivacyLevel.Private)
				&& !expectedEmpty.containsValue(PrivacyLevel.PrivateAggregation)));
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivatePrivateAggregationFineGrained(){
		//Build
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.PrivateAggregation);

		//Execute
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);

		//Assert
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertTrue("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
		assertTrue("Merged constraint should not contain privacy level PrivateAggregation", mergedConstraint.getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation).length == 0);
		Map<DataRange, PrivacyLevel> outputElement1 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,0});
		Map<DataRange, PrivacyLevel> outputElement2 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,1});
		Map<DataRange, PrivacyLevel> outputElement3 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,2});
		assertEquals(1, outputElement1.size());
		assertEquals(1, outputElement2.size());
		assertEquals(1, outputElement3.size());
		assertTrue("Privacy level of element 1 is Private", outputElement1.containsValue(PrivacyLevel.Private));
		assertTrue("Privacy level of element 2 is Private", outputElement2.containsValue(PrivacyLevel.Private));
		assertTrue("Privacy level of element 3 is Private", outputElement3.containsValue(PrivacyLevel.Private));
		Map<DataRange, PrivacyLevel> expectedEmpty = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{2,0}, new long[]{3,2}));
		assertTrue("Any other index has no privacy constraint", expectedEmpty.isEmpty() ||
			(!expectedEmpty.containsValue(PrivacyLevel.Private)
				&& !expectedEmpty.containsValue(PrivacyLevel.PrivateAggregation)));
	}

	@Test
	public void getOperatorTypesRowTest(){
		int rows = 4;
		int cols = 2;
		MatrixBlock m1 = getMatrixBlock(rows, cols);
		OperatorType[] actual = PrivacyPropagator.getOperatorTypesRow(m1);
		OperatorType[] expected = Stream.generate(() -> OperatorType.Aggregate).limit(rows).toArray(OperatorType[]::new);
		assertArrayEquals("All values should be OperatorType.Aggregate", expected, actual);
	}

	@Test
	public void getOperatorTypesRowNonAggTest(){
		int rows = 4;
		int cols = 2;
		int nonAggRow = 2;
		MatrixBlock m1 = getMatrixBlock(rows, cols);
		// Make a single row NNZ=1
		m1.getDenseBlock().set(nonAggRow,0,0);
		OperatorType[] actualArray = PrivacyPropagator.getOperatorTypesRow(m1);
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except one should be OperatorType.Aggregate", expected, actualArray[nonAggRow]);
	}

	@Test
	public void getOperatorTypesRowMultipleNonAggTest(){
		int rows = 4;
		int cols = 2;
		int nonAggRow = 2;
		MatrixBlock m1 = getMatrixBlock(rows, cols);
		// Make two rows NNZ=1
		m1.getDenseBlock().set(nonAggRow,0,0);
		m1.getDenseBlock().set(nonAggRow+1,0,0);
		OperatorType[] actualArray = PrivacyPropagator.getOperatorTypesRow(m1);
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggRow]);
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggRow+1]);
	}

	@Test
	public void getOperatorTypesColTest(){
		int rows = 2;
		int cols = 3;
		MatrixBlock m2 = getMatrixBlock(rows, cols);
		OperatorType[] actual = PrivacyPropagator.getOperatorTypesCol(m2);
		OperatorType[] expected = Stream.generate(() -> OperatorType.Aggregate).limit(cols).toArray(OperatorType[]::new);
		assertArrayEquals("All values should be OperatorType.Aggregate", expected, actual);
	}

	@Test
	public void getOperatorTypesColNonAggTest(){
		int rows = 2;
		int cols = 3;
		int nonAggCol = 1;
		MatrixBlock m2 = getMatrixBlock(rows, cols);
		// Make a single col NNZ=1
		m2.getDenseBlock().set(0,nonAggCol,0);
		OperatorType[] actualArray = PrivacyPropagator.getOperatorTypesCol(m2);
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except one should be OperatorType.Aggregate", expected, actualArray[nonAggCol]);
	}

	@Test
	public void getOperatorTypesColMultipleNonAggTest(){
		int rows = 2;
		int cols = 3;
		int nonAggCol = 1;
		MatrixBlock m2 = getMatrixBlock(rows, cols);
		// Make two cols NNZ=1
		m2.getDenseBlock().set(0,nonAggCol,0);
		m2.getDenseBlock().set(0,nonAggCol+1,0);
		OperatorType[] actualArray = PrivacyPropagator.getOperatorTypesCol(m2);
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggCol]);
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggCol+1]);
	}

	private MatrixBlock getMatrixBlock(int rows, int cols){
		DenseBlock denseM = new DenseBlockLFP64(new int[]{rows,cols});
		for ( int r = 0; r < rows; r++ ){
			for ( int c = 0; c < cols; c++ ){
				denseM.set(r,c,r+c+1);
			}
		}
		return new MatrixBlock(rows,cols,denseM);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAgg(){
		NonAggGeneralizedTest(PrivacyLevel.PrivateAggregation);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivate(){
		NonAggGeneralizedTest(PrivacyLevel.Private);
	}

	private void NonAggGeneralizedTest(PrivacyLevel privacyLevel){
		int nonAggRow = 2;
		MatrixBlock m1 = getMatrixBlock(4,2);
		MatrixBlock m2 = getMatrixBlock(2, 3);
		m1.getDenseBlock().set(nonAggRow,0,0);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().putRow(nonAggRow,2,privacyLevel);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedPrivacyConstraint = PrivacyPropagator.matrixMultiplicationPropagation(m1,constraint1,m2,constraint2);
		Map<DataRange, PrivacyLevel> constraints = mergedPrivacyConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{nonAggRow,0}, new long[]{nonAggRow,1}));
		assertEquals("Output constraint should only contain one privacy level", 1,constraints.size());
		assertEquals(new DataRange(new long[]{2,0},new long[]{2,2}), constraints.keySet().toArray()[0]);
		assertTrue("Output constraints should contain the privacy level " + privacyLevel.toString(),
			constraints.containsValue(privacyLevel));
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAgg2(){
		NonAggGeneralizedColTest(PrivacyLevel.PrivateAggregation);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivate2(){
		NonAggGeneralizedColTest(PrivacyLevel.Private);
	}

	private void NonAggGeneralizedColTest(PrivacyLevel privacyLevel){
		int nonAggCol = 2;
		MatrixBlock m1 = getMatrixBlock(4,2);
		MatrixBlock m2 = getMatrixBlock(2, 3);
		m2.getDenseBlock().set(0,nonAggCol,0);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().putCol(nonAggCol,4,privacyLevel);
		PrivacyConstraint mergedPrivacyConstraint = PrivacyPropagator.matrixMultiplicationPropagation(m1,constraint1,m2,constraint2);
		Map<DataRange, PrivacyLevel> constraints = mergedPrivacyConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{0,nonAggCol}, new long[]{3,nonAggCol}));
		assertEquals("Output constraint should only contain one privacy level", 1,constraints.size());
		assertEquals(new DataRange(new long[]{0,nonAggCol},new long[]{3,nonAggCol}), constraints.keySet().toArray()[0]);
		assertTrue("Output constraints should contain the privacy level " + privacyLevel.toString(),
			constraints.containsValue(privacyLevel));
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColNA(){
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, true);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColNAA(){
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, false);
	}

	private void NonAggGeneralizedRowColTest(PrivacyLevel privacyLevel, boolean putElement){
		int nonAgg = 2;
		MatrixBlock m1 = getMatrixBlock(4,2);
		MatrixBlock m2 = getMatrixBlock(2, 3);
		m1.getDenseBlock().set(nonAgg,0,0);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		if (putElement)
			constraint2.getFineGrainedPrivacy().putElement(0,1, privacyLevel);
		else constraint2.getFineGrainedPrivacy().putCol(1,2,privacyLevel);
		PrivacyConstraint mergedPrivacyConstraint = PrivacyPropagator.matrixMultiplicationPropagation(m1,constraint1,m2,constraint2);
		List<Map.Entry<DataRange, PrivacyLevel>> constraints = mergedPrivacyConstraint.getFineGrainedPrivacy().getAllConstraintsList();
		assertEquals("Output constraint should only contain one privacy level", 1,constraints.size());
		assertEquals(new DataRange(new long[]{2,1},new long[]{2,1}), constraints.get(0).getKey());
		assertEquals("Output constraints should contain the privacy level " + privacyLevel.toString(), privacyLevel,
			constraints.get(0).getValue());
	}
}
