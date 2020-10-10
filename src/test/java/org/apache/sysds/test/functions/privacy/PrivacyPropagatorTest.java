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

import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockLFP64;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.propagation.*;
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
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmGeneralNoFineGrainedGeneralized(constraint1,constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrainedNaive(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmGeneralNoFineGrainedGeneralized(constraint1,constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained2Naive(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained3(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained3Naive(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained4(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained4Naive(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained5(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNoFineGrained5Naive(){
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.Private);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmGeneralNoFineGrainedGeneralized(constraint1, constraint2, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmPropagationPrivateGeneralized(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral2() {
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmPropagationPrivateGeneralized(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralNaive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmPropagationPrivateGeneralized(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral2Naive() {
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmPropagationPrivateGeneralized(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneralPrivateFirstOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		mmPropagationPrivateGeneralized(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral2PrivateFirstOptimized() {
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		mmPropagationPrivateGeneralized(PrivacyLevel.PrivateAggregation, propagator);
	}
	
	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmPropagationTestPrivateFineGrainedGeneralized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrainedNaive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmPropagationTestPrivateFineGrainedGeneralized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrainedPrivateFirstOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		mmPropagationTestPrivateFineGrainedGeneralized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained2(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmPropagationTestPrivateFineGrained2Generalized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained2Naive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmPropagationTestPrivateFineGrained2Generalized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained2PrivateFirstOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		mmPropagationTestPrivateFineGrained2Generalized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivatePrivateAggregationFineGrained(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		mmPropagationTestPrivatePrivateAggregationFineGrainedGeneralized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivatePrivateAggregationFineGrainedNaive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		mmPropagationTestPrivatePrivateAggregationFineGrainedGeneralized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivatePrivateAggregationFineGrainedPrivateFirstOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		mmPropagationTestPrivatePrivateAggregationFineGrainedGeneralized(propagator);
	}

	@Test
	public void getOperatorTypesRowTest(){
		int rows = 4;
		int cols = 2;
		MatrixBlock m1 = getMatrixBlock(rows, cols);
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst(m1, null, null, null);
		OperatorType[] actual = propagator.getOperatorTypesRow();
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
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst(m1, null, null, null);
		OperatorType[] actualArray = propagator.getOperatorTypesRow();
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except one should be OperatorType.Aggregate", expected, actualArray[nonAggRow]);
	}

	@Test
	public void getOperatorTypesRowMultipleNonAggTest(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		getOperatorTypesRowMultipleNonAggTestGeneralized(propagator);
	}

	@Test
	public void getOperatorTypesColTest(){
		int rows = 2;
		int cols = 3;
		MatrixBlock m2 = getMatrixBlock(rows, cols);

		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst(null, null, m2, null);
		OperatorType[] actual = propagator.getOperatorTypesCol();
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
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst(null, null, m2, null);
		OperatorType[] actualArray = propagator.getOperatorTypesCol();
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except one should be OperatorType.Aggregate", expected, actualArray[nonAggCol]);
	}

	@Test
	public void getOperatorTypesColMultipleNonAggTest(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		getOperatorTypesColMultipleNonAggTestGeneralized(propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAgg(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		NonAggGeneralizedTest(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivate(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		NonAggGeneralizedTest(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggNaive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		NonAggGeneralizedTest(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivateNaive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		NonAggGeneralizedTest(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivateOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		NonAggGeneralizedTest(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivatePrivateOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		NonAggGeneralizedTest(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAgg2(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		NonAggGeneralizedColTest(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivate2(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		NonAggGeneralizedColTest(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAgg2Naive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		NonAggGeneralizedColTest(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivate2Naive(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		NonAggGeneralizedColTest(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAgg2PrivateFirstOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		NonAggGeneralizedColTest(PrivacyLevel.PrivateAggregation, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggPrivate2PrivateFirstOptimized(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		NonAggGeneralizedColTest(PrivacyLevel.Private, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColNA(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, true, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColNAA(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirst();
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, false, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColNaiveNA(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, true, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColNaiveNAA(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorNaive();
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, false, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColPrivateFirstOptimizedNA(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, true, propagator);
	}

	@Test
	public void matrixMultiplicationPropagationTestNonAggRowColPrivateFirstOptimizedNAA(){
		MatrixMultiplicationPropagator propagator = new MatrixMultiplicationPropagatorPrivateFirstOptimized();
		NonAggGeneralizedRowColTest(PrivacyLevel.PrivateAggregation, false, propagator);
	}
	
	private static void mmGeneralNoFineGrainedGeneralized(PrivacyConstraint constraint1, PrivacyConstraint constraint2, MatrixMultiplicationPropagator propagator){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		propagator.setFields(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	private static void mmPropagationPrivateGeneralized(PrivacyLevel fineGrainedPrivacyLevel, MatrixMultiplicationPropagator propagator){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3,8},new long[]{2,5}), fineGrainedPrivacyLevel);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		propagator.setFields(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	private static void mmPropagationTestPrivateFineGrainedGeneralized(MatrixMultiplicationPropagator propagator){
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		propagator.setFields(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
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

	private static void mmPropagationTestPrivatePrivateAggregationFineGrainedGeneralized(MatrixMultiplicationPropagator propagator){
		//Build
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.PrivateAggregation);

		//Execute
		propagator.setFields(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();

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

	private static void mmPropagationTestPrivateFineGrained2Generalized(MatrixMultiplicationPropagator propagator){
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		propagator.setFields(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
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

	private static void getOperatorTypesRowMultipleNonAggTestGeneralized(MatrixMultiplicationPropagator propagator){
		int rows = 4;
		int cols = 2;
		int nonAggRow = 2;
		MatrixBlock m1 = getMatrixBlock(rows, cols);
		// Make two rows NNZ=1
		m1.getDenseBlock().set(nonAggRow,0,0);
		m1.getDenseBlock().set(nonAggRow+1,0,0);
		propagator.setFields(m1, null, null, null);
		OperatorType[] actualArray = propagator.getOperatorTypesRow();
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggRow]);
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggRow+1]);
	}

	private static void getOperatorTypesColMultipleNonAggTestGeneralized(MatrixMultiplicationPropagator propagator){
		int rows = 2;
		int cols = 3;
		int nonAggCol = 1;
		MatrixBlock m2 = getMatrixBlock(rows, cols);
		// Make two cols NNZ=1
		m2.getDenseBlock().set(0,nonAggCol,0);
		m2.getDenseBlock().set(0,nonAggCol+1,0);
		propagator.setFields(null, null, m2, null);
		OperatorType[] actualArray = propagator.getOperatorTypesCol();
		OperatorType expected = OperatorType.NonAggregate;
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggCol]);
		assertEquals("All values except two should be OperatorType.Aggregate", expected, actualArray[nonAggCol+1]);
	}
	
	private static MatrixBlock getMatrixBlock(int rows, int cols){
		DenseBlock denseM = new DenseBlockLFP64(new int[]{rows,cols});
		for ( int r = 0; r < rows; r++ ){
			for ( int c = 0; c < cols; c++ ){
				denseM.set(r,c,r+c+1);
			}
		}
		return new MatrixBlock(rows,cols,denseM);
	}

	private static void NonAggGeneralizedTest(PrivacyLevel privacyLevel, MatrixMultiplicationPropagator propagator){
		int nonAggRow = 2;
		MatrixBlock m1 = getMatrixBlock(4,2);
		MatrixBlock m2 = getMatrixBlock(2, 3);
		m1.getDenseBlock().set(nonAggRow,0,0);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().putRow(nonAggRow,2,privacyLevel);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		propagator.setFields(m1, constraint1, m2, constraint2);
		PrivacyConstraint mergedPrivacyConstraint = propagator.propagate();
		Map<DataRange, PrivacyLevel> constraints = mergedPrivacyConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{nonAggRow,0}, new long[]{nonAggRow,1}));
		assertTrue("Output constraints should contain the privacy level " + privacyLevel.toString(),
			constraints.containsValue(privacyLevel));
		if ( privacyLevel == PrivacyLevel.Private)
			assertFalse("Output constraints should not contain the privacy level PrivateAggregation",
				constraints.containsValue(PrivacyLevel.PrivateAggregation));
		else if ( privacyLevel == PrivacyLevel.PrivateAggregation )
			assertFalse("Output constraints should not contain the privacy level Private",
				constraints.containsValue(PrivacyLevel.Private));
	}
	
	private static void NonAggGeneralizedColTest(PrivacyLevel privacyLevel, MatrixMultiplicationPropagator propagator){
		int nonAggCol = 2;
		MatrixBlock m1 = getMatrixBlock(4,2);
		MatrixBlock m2 = getMatrixBlock(2, 3);
		m2.getDenseBlock().set(0,nonAggCol,0);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().putCol(nonAggCol,4,privacyLevel);
		propagator.setFields(m1, constraint1, m2, constraint2);
		PrivacyConstraint mergedPrivacyConstraint = propagator.propagate();
		Map<DataRange, PrivacyLevel> constraints = mergedPrivacyConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{0,nonAggCol}, new long[]{3,nonAggCol}));
		assertTrue("Output constraints should contain the privacy level " + privacyLevel.toString(),
			constraints.containsValue(privacyLevel));
		if ( privacyLevel == PrivacyLevel.Private)
			assertFalse("Output constraints should not contain the privacy level PrivateAggregation",
				constraints.containsValue(PrivacyLevel.PrivateAggregation));
		else if ( privacyLevel == PrivacyLevel.PrivateAggregation )
			assertFalse("Output constraints should not contain the privacy level Private",
				constraints.containsValue(PrivacyLevel.Private));
	}
	
	private static void NonAggGeneralizedRowColTest(PrivacyLevel privacyLevel, boolean putElement, MatrixMultiplicationPropagator propagator){
		int nonAgg = 2;
		MatrixBlock m1 = getMatrixBlock(4,2);
		MatrixBlock m2 = getMatrixBlock(2, 3);
		m1.getDenseBlock().set(nonAgg,0,0);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		if (putElement)
			constraint2.getFineGrainedPrivacy().putElement(0,1, privacyLevel);
		else constraint2.getFineGrainedPrivacy().putCol(1,2,privacyLevel);
		propagator.setFields(m1, constraint1, m2, constraint2);
		PrivacyConstraint mergedPrivacyConstraint = propagator.propagate();

		// Check output
		List<Map.Entry<DataRange, PrivacyLevel>> constraints = mergedPrivacyConstraint.getFineGrainedPrivacy().getAllConstraintsList();
		int privacyLevelSum = 0;
		DataRange levelRange = null;
		PrivacyLevel level = PrivacyLevel.None;
		for ( Map.Entry<DataRange, PrivacyLevel> constraint : constraints )
			if ( constraint.getValue() == privacyLevel ){
				privacyLevelSum++;
				levelRange = (DataRange)constraint.getKey();
				level = (PrivacyLevel) constraint.getValue();
			}

		assertEquals("Output constraint should only contain one privacy level which is not none", 1,privacyLevelSum);
		assertEquals(new DataRange(new long[]{2,1},new long[]{2,1}), levelRange);
		assertEquals("Output constraints should contain the privacy level " + privacyLevel.toString(), privacyLevel,
			level);
	}
}
