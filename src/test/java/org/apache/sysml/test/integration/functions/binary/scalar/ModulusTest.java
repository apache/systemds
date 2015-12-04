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

package org.apache.sysml.test.integration.functions.binary.scalar;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ModulusTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/binary/scalar/";
	
	private double intIntValue1 = 9;
	private double intIntValue2 = 4;
	private double intDoubleValue1 = 9;
	private double intDoubleValue2 = 4;
	private double doubleDoubleValue1 = 9;
	private double doubleDoubleValue2 = 4;
	private double doubleIntValue1 = 9;
	private double doubleIntValue2 = 4;
	
	private double computedIntIntValue = intIntValue1 % intIntValue2;
	private double computedIntDoubleValue = intDoubleValue1 % intDoubleValue2;
	private double computedDoubleDoubleValue = doubleDoubleValue1 % doubleDoubleValue2;
	private double computedDoubleIntValue = doubleIntValue1 % doubleIntValue2;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("ConstConstTest", new TestConfiguration(TEST_DIR, "ModulusTest", new String[] {
				"int_int", "int_double", "double_double", "double_int" }));
		addTestConfiguration("VarConstTest", new TestConfiguration(TEST_DIR, "ModulusTest", new String[] { "int_int",
				"int_double", "double_double", "double_int" }));
		addTestConfiguration("ConstVarTest", new TestConfiguration(TEST_DIR, "ModulusTest", new String[] { "int_int",
				"int_double", "double_double", "double_int" }));
		addTestConfiguration("VarVarTest", new TestConfiguration(TEST_DIR, "ModulusTest", new String[] { "int_int",
				"int_double", "double_double", "double_int" }));
		addTestConfiguration("PositiveDivisionByZeroTest", new TestConfiguration(TEST_DIR, "ModulusSingleTest",
				new String[] { "computed" }));
		addTestConfiguration("NegativeDivisionByZeroTest", new TestConfiguration(TEST_DIR, "ModulusSingleTest",
				new String[] { "computed" }));
		addTestConfiguration("ZeroDivisionByZeroTest", new TestConfiguration(TEST_DIR, "ModulusSingleTest",
				new String[] { "computed" }));		
	}

	@Test
	public void testConstConst() {

		TestConfiguration config = availableTestConfigurations.get("ConstConstTest");
		
		config.addVariable("intintvardeclaration", "");
		config.addVariable("intintop1", intIntValue1);
		config.addVariable("intintop2", intIntValue2);
		config.addVariable("intdoublevardeclaration", "");
		config.addVariable("intdoubleop1", intDoubleValue1);
		config.addVariable("intdoubleop2", intDoubleValue2);
		config.addVariable("doubledoublevardeclaration", "");
		config.addVariable("doubledoubleop1", doubleDoubleValue1);
		config.addVariable("doubledoubleop2", doubleDoubleValue2);
		config.addVariable("doubleintvardeclaration", "");
		config.addVariable("doubleintop1", doubleIntValue1);
		config.addVariable("doubleintop2", doubleIntValue2);

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}
	
	@Test
	public void testVarConst() {

		TestConfiguration config = availableTestConfigurations.get("VarConstTest");
		config.addVariable("intintvardeclaration", "IntIntVar = " + intIntValue1 + ";");
		config.addVariable("intintop1", "IntIntVar");
		config.addVariable("intintop2", intIntValue2);
		config.addVariable("intdoublevardeclaration", "IntDoubleVar = " + intDoubleValue1 + ";");
		config.addVariable("intdoubleop1", "IntDoubleVar");
		config.addVariable("intdoubleop2", intDoubleValue2);
		config.addVariable("doubledoublevardeclaration", "DoubleDoubleVar = " + doubleDoubleValue1 + ";");
		config.addVariable("doubledoubleop1", "DoubleDoubleVar");
		config.addVariable("doubledoubleop2", doubleDoubleValue2);
		config.addVariable("doubleintvardeclaration", "DoubleIntVar = " + doubleIntValue1 + ";");
		config.addVariable("doubleintop1", "DoubleIntVar");
		config.addVariable("doubleintop2", doubleIntValue2);

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testConstVar() {

		TestConfiguration config = availableTestConfigurations.get("ConstVarTest");
		config.addVariable("intintvardeclaration", "IntIntVar = " + intIntValue2 + ";");
		config.addVariable("intintop1", intIntValue1);
		config.addVariable("intintop2", "IntIntVar");
		config.addVariable("intdoublevardeclaration", "IntDoubleVar = " + intDoubleValue2 + ";");
		config.addVariable("intdoubleop1", intDoubleValue1);
		config.addVariable("intdoubleop2", "IntDoubleVar");
		config.addVariable("doubledoublevardeclaration", "DoubleDoubleVar = " + doubleDoubleValue2 + ";");
		config.addVariable("doubledoubleop1", doubleDoubleValue1);
		config.addVariable("doubledoubleop2", "DoubleDoubleVar");
		config.addVariable("doubleintvardeclaration", "DoubleIntVar = " + doubleIntValue2 + ";");
		config.addVariable("doubleintop1", doubleIntValue1);
		config.addVariable("doubleintop2", "DoubleIntVar");

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testVarVar() {

		TestConfiguration config = availableTestConfigurations.get("VarVarTest");
		config.addVariable("intintvardeclaration", "IntIntVar1 = " + intIntValue1 + ";" + "IntIntVar2 = "
				+ intIntValue2 + ";");
		config.addVariable("intintop1", "IntIntVar1");
		config.addVariable("intintop2", "IntIntVar2");
		config.addVariable("intdoublevardeclaration", "IntDoubleVar1 = " + intDoubleValue1 + ";" + "IntDoubleVar2 = "
				+ intDoubleValue2 + ";");
		config.addVariable("intdoubleop1", "IntDoubleVar1");
		config.addVariable("intdoubleop2", "IntDoubleVar2");
		config.addVariable("doubledoublevardeclaration", "DoubleDoubleVar1 = " + doubleDoubleValue1 + ";"
				+ "DoubleDoubleVar2 = " + doubleDoubleValue2 + ";");
		config.addVariable("doubledoubleop1", "DoubleDoubleVar1");
		config.addVariable("doubledoubleop2", "DoubleDoubleVar2");
		config.addVariable("doubleintvardeclaration", "DoubleIntVar1 = " + doubleIntValue1 + ";" + "DoubleIntVar2 = "
				+ doubleIntValue2 + ";");
		config.addVariable("doubleintop1", "DoubleIntVar1");
		config.addVariable("doubleintop2", "DoubleIntVar2");

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}	
	
	@Test
	public void testPositiveIntegerDivisionByZero() {
		double op1 = 5;
		double op2 = 0;

		TestConfiguration config = availableTestConfigurations.get("PositiveDivisionByZeroTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", op1);
		config.addVariable("op2", op2);

		loadTestConfiguration(config);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("computed", Double.NaN);

		runTest();
		
		compareResults();
	}

	@Test
	public void testPositiveDoubleDivisionByZero() {
		double op1 = 5;
		double op2 = 0;

		TestConfiguration config = availableTestConfigurations.get("PositiveDivisionByZeroTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", op1);
		config.addVariable("op2", op2);

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("computed", Double.NaN);

		runTest();

		compareResults();
	}

	@Test
	public void testNegativeDoubleDivisionByZero() {
		double op1 = -5;
		double op2 = 0;

		TestConfiguration config = availableTestConfigurations.get("NegativeDivisionByZeroTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", op1);
		config.addVariable("op2", op2);

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("computed", Double.NaN);

		runTest();

		compareResults();
	}

	@Test
	public void testNegativeIntegerDivisionByZero() {
		double op1 = -5;
		double op2 = 0;

		TestConfiguration config = availableTestConfigurations.get("NegativeDivisionByZeroTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", op1);
		config.addVariable("op2", op2);

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("computed", Double.NaN);

		runTest();

		compareResults();
	}

	@Test
	public void testZeroDoubleDivisionByZero() {
		double op1 = 0;
		double op2 = 0;

		TestConfiguration config = availableTestConfigurations.get("ZeroDivisionByZeroTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", op1);
		config.addVariable("op2", op2);

		loadTestConfiguration(config);

		createHelperMatrix();
		writeExpectedHelperMatrix("computed", Double.NaN);

		runTest();

		compareResults();
	}

	@Test
	public void testZeroIntegerDivisionByZero() {
		double op1 = 0;
		double op2 = 0;

		TestConfiguration config = availableTestConfigurations.get("ZeroDivisionByZeroTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", op1);
		config.addVariable("op2", op2);

		loadTestConfiguration(config);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("computed", Double.NaN);

		runTest();
		
		compareResults();
	}	
}
