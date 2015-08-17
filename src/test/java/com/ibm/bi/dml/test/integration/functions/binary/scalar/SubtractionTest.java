/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class SubtractionTest extends AutomatedTestBase
{

	
	private static final String TEST_DIR = "functions/binary/scalar/";
	
	@Override
	public void setUp() {
		addTestConfiguration("ConstConstTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("VarConstTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("ConstVarTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("VarVarTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("NegativeConstConstTest", new TestConfiguration(TEST_DIR, "SubtractionTest",
				new String[] { "int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("NegativeVarConstTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("NegativeConstVarTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("NegativeVarVarTest", new TestConfiguration(TEST_DIR, "SubtractionTest", new String[] {
				"int_int", "int_double", "double_double", "double_double" }));
		addTestConfiguration("ConstConstConstTest", new TestConfiguration(TEST_DIR, "SubtractionMultipleOperantsTest",
				new String[] { "int_int_int", "double_double_double" }));
	}

	@Test
	public void testTwoMinusOne() {
		int intIntValue1 = 2;
		int intIntValue2 = 1;
		int intDoubleValue1 = 2;
		double intDoubleValue2 = 1;
		double doubleDoubleValue1 = 2;
		double doubleDoubleValue2 = 1;
		double doubleIntValue1 = 2;
		int doubleIntValue2 = 1;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testConstConst() {
		int intIntValue1 = 3;
		int intIntValue2 = 2;
		int intDoubleValue1 = 3;
		double intDoubleValue2 = 2;
		double doubleDoubleValue1 = 3;
		double doubleDoubleValue2 = 2;
		double doubleIntValue1 = 3;
		int doubleIntValue2 = 2;

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

		// loadTestConfiguration("ConstConstTest");
		loadTestConfiguration(config);

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

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
		int intIntValue1 = 3;
		int intIntValue2 = 2;
		int intDoubleValue1 = 3;
		double intDoubleValue2 = 2;
		double doubleDoubleValue1 = 3;
		double doubleDoubleValue2 = 2;
		double doubleIntValue1 = 3;
		int doubleIntValue2 = 2;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

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
		int intIntValue1 = 3;
		int intIntValue2 = 2;
		int intDoubleValue1 = 3;
		double intDoubleValue2 = 2;
		double doubleDoubleValue1 = 3;
		double doubleDoubleValue2 = 2;
		double doubleIntValue1 = 3;
		int doubleIntValue2 = 2;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

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
		int intIntValue1 = 3;
		int intIntValue2 = 2;
		int intDoubleValue1 = 3;
		double intDoubleValue2 = 2;
		double doubleDoubleValue1 = 3;
		double doubleDoubleValue2 = 2;
		double doubleIntValue1 = 3;
		int doubleIntValue2 = 2;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testNegativeConstConst() {
		int intIntValue1 = 2;
		int intIntValue2 = 3;
		int intDoubleValue1 = 2;
		double intDoubleValue2 = 3;
		double doubleDoubleValue1 = 2;
		double doubleDoubleValue2 = 3;
		double doubleIntValue1 = 2;
		int doubleIntValue2 = 3;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testNegativeVarConst() {
		int intIntValue1 = 2;
		int intIntValue2 = 3;
		int intDoubleValue1 = 2;
		double intDoubleValue2 = 3;
		double doubleDoubleValue1 = 2;
		double doubleDoubleValue2 = 3;
		double doubleIntValue1 = 2;
		int doubleIntValue2 = 3;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testNegativeConstVar() {
		int intIntValue1 = 2;
		int intIntValue2 = 3;
		int intDoubleValue1 = 2;
		double intDoubleValue2 = 3;
		double doubleDoubleValue1 = 2;
		double doubleDoubleValue2 = 3;
		double doubleIntValue1 = 2;
		int doubleIntValue2 = 3;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testNegativeVarVar() {
		int intIntValue1 = 2;
		int intIntValue2 = 3;
		int intDoubleValue1 = 2;
		double intDoubleValue2 = 3;
		double doubleDoubleValue1 = 2;
		double doubleDoubleValue2 = 3;
		double doubleIntValue1 = 2;
		int doubleIntValue2 = 3;

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

		double computedIntIntValue = intIntValue1 - intIntValue2;
		double computedIntDoubleValue = intDoubleValue1 - intDoubleValue2;
		double computedDoubleDoubleValue = doubleDoubleValue1 - doubleDoubleValue2;
		double computedDoubleIntValue = doubleIntValue1 - doubleIntValue2;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);

		runTest();

		compareResults();
	}

	@Test
	public void testConstConstConst() {
		int intIntIntValue1 = 3;
		int intIntIntValue2 = 4;
		int intIntIntValue3 = 5;
		double doubleDoubleDoubleValue1 = 3;
		double doubleDoubleDoubleValue2 = 4;
		double doubleDoubleDoubleValue3 = 5;

		TestConfiguration config = availableTestConfigurations.get("ConstConstConstTest");
		config.addVariable("intintintop1", intIntIntValue1);
		config.addVariable("intintintop2", intIntIntValue2);
		config.addVariable("intintintop3", intIntIntValue3);
		config.addVariable("doubledoubledoubleop1", doubleDoubleDoubleValue1);
		config.addVariable("doubledoubledoubleop2", doubleDoubleDoubleValue2);
		config.addVariable("doubledoubledoubleop3", doubleDoubleDoubleValue3);

		loadTestConfiguration(config);

		double computedIntIntIntValue = intIntIntValue1 - intIntIntValue2 - intIntIntValue3;
		double computedDoubleDoubleDoubleValue = doubleDoubleDoubleValue1 - doubleDoubleDoubleValue2
				- doubleDoubleDoubleValue3;

		createHelperMatrix();
		writeExpectedHelperMatrix("int_int_int", computedIntIntIntValue);
		writeExpectedHelperMatrix("double_double_double", computedDoubleDoubleDoubleValue);

		runTest();

		compareResults();
	}

}
