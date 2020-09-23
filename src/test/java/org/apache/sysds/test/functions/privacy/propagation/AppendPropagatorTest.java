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

package org.apache.sysds.test.functions.privacy.propagation;

import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.propagation.*;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class AppendPropagatorTest extends AutomatedTestBase {

	@Override public void setUp() {

	}

	@Test
	public void generalOnlyRBindPrivate1Test(){
		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyRBindPrivateAggregation1Test(){
		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyRBindNoneTest(){
		generalOnlyRBindTest(new PrivacyConstraint(), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyRBindPrivate2Test(){
		generalOnlyRBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private));
	}

	@Test
	public void generalOnlyRBindPrivateAggregation2Test(){
		generalOnlyRBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
	}

	@Test
	public void generalOnlyRBindPrivatePrivateTest(){
		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private));
	}

	@Test
	public void generalOnlyRBindPrivatePrivateAggregationTest(){
		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
	}

	private void generalOnlyRBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
		int columns = 2;
		int rows1 = 4;
		int rows2 = 3;
		MatrixBlock inputMatrix1 = new MatrixBlock(rows1,columns,3);
		MatrixBlock inputMatrix2 = new MatrixBlock(rows2,columns,4);
		AppendPropagator propagator = new RBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0,0}, new long[]{rows1-1,columns-1}));
		firstHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint1.getPrivacyLevel(),level));
		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{rows1,0}, new long[]{rows1+rows2-1,columns-1}));
		secondHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint2.getPrivacyLevel(),level));
	}

	@Test
	public void generalOnlyCBindPrivate1Test(){
		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyCBindPrivateAggregation1Test(){
		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyCBindNoneTest(){
		generalOnlyCBindTest(new PrivacyConstraint(), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyCBindPrivate2Test(){
		generalOnlyCBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private));
	}

	@Test
	public void generalOnlyCBindPrivateAggregation2Test(){
		generalOnlyCBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
	}

	@Test
	public void generalOnlyCBindPrivatePrivateTest(){
		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private));
	}

	@Test
	public void generalOnlyCBindPrivatePrivateAggregationTest(){
		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
	}

	private void generalOnlyCBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
		int rows = 2;
		int columns1 = 4;
		int columns2 = 3;
		MatrixBlock inputMatrix1 = new MatrixBlock(rows,columns1,3);
		MatrixBlock inputMatrix2 = new MatrixBlock(rows,columns2,4);
		AppendPropagator propagator = new CBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0,0}, new long[]{rows-1,columns1-1}));
		firstHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint1.getPrivacyLevel(),level));
		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0,columns1}, new long[]{rows,columns1+columns2-1}));
		secondHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint2.getPrivacyLevel(),level));
	}

	@Test
	public void generalOnlyListAppendPrivate1Test(){
		generalOnlyListAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyListAppendPrivateAggregation1Test(){
		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyListAppendNoneTest(){
		generalOnlyListAppendTest(new PrivacyConstraint(), new PrivacyConstraint());
	}

	@Test
	public void generalOnlyListAppendPrivate2Test(){
		generalOnlyListAppendTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private));
	}

	@Test
	public void generalOnlyListAppendPrivateAggregation2Test(){
		generalOnlyListAppendTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
	}

	@Test
	public void generalOnlyListAppendPrivatePrivateTest(){
		generalOnlyListAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private));
	}

	@Test
	public void generalOnlyListAppendPrivatePrivateAggregationTest(){
		generalOnlyListAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
	}

	private void generalOnlyListAppendTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
		int length1 = 6;
		List<Data> dataList1 = Arrays.asList(new Data[length1]);
		ListObject input1 = new ListObject(dataList1);
		int length2 = 11;
		List<Data> dataList2 = Arrays.asList(new Data[length2]);
		ListObject input2 = new ListObject(dataList2);
		Propagator propagator = new ListAppendPropagator(input1, constraint1, input2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0}, new long[]{length1-1})
		);
		firstHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint1.getPrivacyLevel(),level));
		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[length1], new long[]{length1+length2-1})
		);
		secondHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint2.getPrivacyLevel(),level));
	}

	@Test
	public void finegrainedRBindPrivate1(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		finegrainedRBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedRBindPrivateAndPrivateAggregate1(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		finegrainedRBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedRBindPrivate2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		finegrainedRBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedRBindPrivateAndPrivateAggregate2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
		finegrainedRBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedRBindPrivate12(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		finegrainedRBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedRBindPrivateAndPrivateAggregate12(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{2,0}), PrivacyLevel.PrivateAggregation);
		finegrainedRBindTest(constraint1, constraint2);
	}

	private void finegrainedRBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
		int columns = 2;
		int rows1 = 4;
		int rows2 = 3;
		MatrixBlock inputMatrix1 = new MatrixBlock(rows1,columns,3);
		MatrixBlock inputMatrix2 = new MatrixBlock(rows2,columns,4);
		AppendPropagator propagator = new RBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0,0}, new long[]{rows1-1,columns-1}));
		constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
				firstHalfPrivacy.containsValue(constraint.getValue()))
		);
		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{rows1,0}, new long[]{rows1+rows2-1,columns-1}));
		constraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 2",
				secondHalfPrivacy.containsValue(constraint.getValue()))
		);
	}

	@Test
	public void finegrainedCBindPrivate1(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		finegrainedCBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedCBindPrivateAndPrivateAggregate1(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		finegrainedCBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedCBindPrivate2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		finegrainedCBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedCBindPrivateAndPrivateAggregate2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
		finegrainedCBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedCBindPrivate12(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		finegrainedCBindTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedCBindPrivateAndPrivateAggregate12(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{2,0}), PrivacyLevel.PrivateAggregation);
		finegrainedCBindTest(constraint1, constraint2);
	}

	private void finegrainedCBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
		int rows = 6;
		int columns1 = 4;
		int columns2 = 3;
		MatrixBlock inputMatrix1 = new MatrixBlock(rows,columns1,3);
		MatrixBlock inputMatrix2 = new MatrixBlock(rows,columns2,4);
		AppendPropagator propagator = new CBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0,0}, new long[]{rows-1,columns1-1}));
		constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
				firstHalfPrivacy.containsValue(constraint.getValue()))
		);
		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0,columns1}, new long[]{rows,columns1+columns2-1}));
		constraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 2",
				secondHalfPrivacy.containsValue(constraint.getValue()))
		);
	}

	@Test
	public void finegrainedListAppendPrivate1(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		finegrainedListAppendTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedListAppendPrivateAndPrivateAggregate1(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		finegrainedListAppendTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedListAppendPrivate2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
		finegrainedListAppendTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedListAppendPrivateAndPrivateAggregate2(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
		finegrainedListAppendTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedListAppendPrivate12(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{4}), PrivacyLevel.Private);
		finegrainedListAppendTest(constraint1, constraint2);
	}

	@Test
	public void finegrainedListAppendPrivateAndPrivateAggregate12(){
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0},new long[]{0}), PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2},new long[]{3}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0},new long[]{1}), PrivacyLevel.Private);
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
		finegrainedListAppendTest(constraint1, constraint2);
	}

	private void finegrainedListAppendTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
		int length1 = 6;
		List<Data> dataList1 = Arrays.asList(new Data[length1]);
		ListObject input1 = new ListObject(dataList1);
		int length2 = 11;
		List<Data> dataList2 = Arrays.asList(new Data[length2]);
		ListObject input2 = new ListObject(dataList2);
		Propagator propagator = new ListAppendPropagator(input1, constraint1, input2, constraint2);
		PrivacyConstraint mergedConstraint = propagator.propagate();
		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{0}, new long[]{length1-1})
		);
		constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
				firstHalfPrivacy.containsValue(constraint.getValue()))
		);
		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
			new DataRange(new long[]{length1}, new long[]{length1+length2-1})
		);
		constraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 2",
				secondHalfPrivacy.containsValue(constraint.getValue()))
		);
	}
}
