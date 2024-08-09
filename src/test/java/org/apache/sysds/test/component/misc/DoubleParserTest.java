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

package org.apache.sysds.test.component.misc;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.utils.DoubleParser;
import org.junit.Test;

public class DoubleParserTest {

	@Test
	public void testDoubleParser() {
		compareToDoubleParser("0.01324");
	}

	@Test
	public void testDoubleParser_appending_zero() {
		String s = "0.01324";
		for(int i = 0; i < 100; i++) {
			s = s + "0";
			compareToDoubleParser(s);
		}
	}

	@Test(expected = Exception.class)
	public void invalid() {
		parse("Hi");
	}

	@Test(expected = Exception.class)
	public void invalid2() {
		parse("0ds");
	}

	@Test(expected = Exception.class)
	public void invalid3() {
		parse("0da");
	}

	@Test
	public void parseVeryLongDouble() {
		compareToDoubleParser(
			"10285923939581294294521123123194523123275123123123.132231241512323782173921839313223214521513417239817298372198739817329813");
		// way to long....
	}

	@Test
	public void parseNegative() {
		compareToDoubleParser("-3122414.13");
	}

	@Test
	public void parsePositive() {
		compareToDoubleParser("+3122414.13");
	}

	@Test
	public void parseInf() {
		compareToDoubleParser("Infinity");
	}

	@Test
	public void parseNegInf() {
		compareToDoubleParser("-Infinity");
	}

	@Test
	public void parseNan() {
		compareToDoubleParser("NaN");
	}

	@Test
	public void parseExponent() {
		compareToDoubleParser("132e32");
	}

	@Test
	public void parseNegativeExponent() {
		compareToDoubleParser("-132e32");
	}

	@Test
	public void parseNegativeExponent2() {
		compareToDoubleParser("-132e-32");
	}

	@Test
	public void parsePostiveExponent2() {
		compareToDoubleParser("-132e+32");
	}

	@Test(expected = Exception.class)
	public void parseAlmostExponentButNotQuite() {
		parse("-132e-3e2");
	}

	@Test(expected = Exception.class)
	public void parseAlmostExponentButNotQuite2() {
		parse("-132ee-3e2");
	}

	@Test
	public void toLargeExponent() {
		compareToDoubleParser("-132e32323141422");
	}

	@Test
	public void parseVerySmall() {
		compareToDoubleParser("0.0000000000000000001423");
	}

	@Test
	public void parseVeryLarge() {
		compareToDoubleParser("10233333332414251432323230.0");
	}

	@Test
	public void parseVeryLarge2() {
		compareToDoubleParser("99999999999999999.0");
	}

	@Test
	public void parseVeryLarge3() {
		compareToDoubleParser("999999999999999999999999999999999999999999.0");
	}

	@Test
	public void exponentWithCapitalLetter() {
		compareToDoubleParser("-132E32");
	}

	@Test
	public void parseWithWhitespace() {
		compareToDoubleParser("   132.14");
	}

	@Test 
	public void parsePowerOf10(){
		compareToDoubleParser("132e10");
	}

	@Test 
	public void parsePowerOfMinus22(){
		compareToDoubleParser("132e-22");
	}

	@Test(expected = Exception.class)
	public void parseErrorWithMultipleDots() {
		parse("132.14.13");
	}

	@Test(expected = Exception.class)
	public void parseWhitespace() {
		parse("      ");
	}

	private void compareToDoubleParser(String s) {
		assertEquals(Double.parseDouble(s), parse(s), 0.00);
	}

	private double parse(String s) {
		return DoubleParser.parseFloatingPointLiteral(s, 0, s.length());
	}
}
