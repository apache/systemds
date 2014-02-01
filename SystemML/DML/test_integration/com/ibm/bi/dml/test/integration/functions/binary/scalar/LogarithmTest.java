/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class LogarithmTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static final double EPS = 1e-14;
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/scalar/";
		availableTestConfigurations.put("ConstConstTest", new TestConfiguration("LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
		availableTestConfigurations.put("VarConstTest", new TestConfiguration("LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
		availableTestConfigurations.put("ConstVarTest", new TestConfiguration("LogarithmTest",
				new String[] { "int_int", "int_double", "double_double", "double_double"}));
		availableTestConfigurations.put("VarVarTest", new TestConfiguration("LogarithmTest",
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
		
		loadTestConfiguration("ConstConstTest");
		
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
		
		loadTestConfiguration("VarConstTest");
		
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
		
		loadTestConfiguration("ConstVarTest");
		
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
		
		loadTestConfiguration("VarVarTest");
		
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
