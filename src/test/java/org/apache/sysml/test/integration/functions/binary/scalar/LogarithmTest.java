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

package org.apache.sysml.test.integration.functions.binary.scalar;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



public class LogarithmTest extends AutomatedTestBase 
{
		
	private static final String TEST_DIR = "functions/binary/scalar/";
	private static final double EPS = 1e-14;
	
	@Override
	public void setUp() {
		addTestConfiguration("ConstConstTest", new TestConfiguration(TEST_DIR, "LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
		addTestConfiguration("VarConstTest", new TestConfiguration(TEST_DIR, "LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
		addTestConfiguration("ConstVarTest", new TestConfiguration(TEST_DIR, "LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
		addTestConfiguration("VarVarTest", new TestConfiguration(TEST_DIR, "LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
	}
	
	@Test
	public void testConstConst() {
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
		
		double computedIntIntValue = Math.log(intIntValue1) / Math.log(intIntValue2);
		double computedIntDoubleValue = Math.log(intDoubleValue1) / Math.log(intDoubleValue2);
		double computedDoubleDoubleValue = Math.log(doubleDoubleValue1) / Math.log(doubleDoubleValue2);
		double computedDoubleIntValue = Math.log(doubleIntValue1) / Math.log(doubleIntValue2);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);
		
		runTest();
		
		compareResults(EPS);
	}
	
	@Test
	public void testVarConst() {
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
		
		double computedIntIntValue = Math.log(intIntValue1) / Math.log(intIntValue2);
		double computedIntDoubleValue = Math.log(intDoubleValue1) / Math.log(intDoubleValue2);
		double computedDoubleDoubleValue = Math.log(doubleDoubleValue1) / Math.log(doubleDoubleValue2);
		double computedDoubleIntValue = Math.log(doubleIntValue1) / Math.log(doubleIntValue2);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);
		
		runTest();
		
		compareResults(EPS);
	}
	
	@Test
	public void testConstVar() {
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
		
		double computedIntIntValue = Math.log(intIntValue1) / Math.log(intIntValue2);
		double computedIntDoubleValue = Math.log(intDoubleValue1) / Math.log(intDoubleValue2);
		double computedDoubleDoubleValue = Math.log(doubleDoubleValue1) / Math.log(doubleDoubleValue2);
		double computedDoubleIntValue = Math.log(doubleIntValue1) / Math.log(doubleIntValue2);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);
		
		runTest();
		
		compareResults(EPS);
	}
	
	@Test
	public void testVarVar() {
		int intIntValue1 = 2;
		int intIntValue2 = 3;
		int intDoubleValue1 = 2;
		double intDoubleValue2 = 3;
		double doubleDoubleValue1 = 2;
		double doubleDoubleValue2 = 3;
		double doubleIntValue1 = 2;
		int doubleIntValue2 = 3;
		
		TestConfiguration config = availableTestConfigurations.get("VarVarTest");
		config.addVariable("intintvardeclaration", "IntIntVar1 = " + intIntValue1 + ";" +
				"IntIntVar2 = " + intIntValue2 + ";");
		config.addVariable("intintop1", "IntIntVar1");
		config.addVariable("intintop2", "IntIntVar2");
		config.addVariable("intdoublevardeclaration", "IntDoubleVar1 = " + intDoubleValue1 + ";" +
				"IntDoubleVar2 = " + intDoubleValue2 + ";");
		config.addVariable("intdoubleop1", "IntDoubleVar1");
		config.addVariable("intdoubleop2", "IntDoubleVar2");
		config.addVariable("doubledoublevardeclaration", "DoubleDoubleVar1 = " + doubleDoubleValue1 + ";" +
				"DoubleDoubleVar2 = " + doubleDoubleValue2 + ";");
		config.addVariable("doubledoubleop1", "DoubleDoubleVar1");
		config.addVariable("doubledoubleop2", "DoubleDoubleVar2");
		config.addVariable("doubleintvardeclaration", "DoubleIntVar1 = " + doubleIntValue1 + ";" +
				"DoubleIntVar2 = " + doubleIntValue2 + ";");
		config.addVariable("doubleintop1", "DoubleIntVar1");
		config.addVariable("doubleintop2", "DoubleIntVar2");
		
		loadTestConfiguration(config);
		
		double computedIntIntValue = Math.log(intIntValue1) / Math.log(intIntValue2);
		double computedIntDoubleValue = Math.log(intDoubleValue1) / Math.log(intDoubleValue2);
		double computedDoubleDoubleValue = Math.log(doubleDoubleValue1) / Math.log(doubleDoubleValue2);
		double computedDoubleIntValue = Math.log(doubleIntValue1) / Math.log(doubleIntValue2);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int_int", computedIntIntValue);
		writeExpectedHelperMatrix("int_double", computedIntDoubleValue);
		writeExpectedHelperMatrix("double_double", computedDoubleDoubleValue);
		writeExpectedHelperMatrix("double_int", computedDoubleIntValue);
		
		runTest();
		
		compareResults(EPS);
	}

}
