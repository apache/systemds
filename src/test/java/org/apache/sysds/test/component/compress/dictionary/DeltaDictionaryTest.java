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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.junit.Assert;
import org.junit.Test;

public class DeltaDictionaryTest {

	@Test
	public void testScalarOpRightMultiplySingleColumn() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2}, 1);
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightMultiplyTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4, 6, 8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testNegScalarOpRightMultiplyTwoColumns() {
		double scalar = -2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {-2, -4, -6, -8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpLeftMultiplyTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4, 6, 8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightDivideTwoColumns() {
		double scalar = 0.5;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Divide.getDivideFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4, 6, 8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightPlusSingleColumn() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2}, 1);
		ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {3, 2};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightPlusTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {3, 4, 3, 4};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightMinusTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Minus.getMinusFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {-1, 0, 3, 4};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpLeftPlusTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new LeftScalarOperator(Plus.getPlusFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {3, 4, 3, 4};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test(expected = NotImplementedException.class)
	public void testNotImplemented() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new LeftScalarOperator(And.getAndFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
	}
}
