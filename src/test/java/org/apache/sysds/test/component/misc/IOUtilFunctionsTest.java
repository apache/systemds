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

import static org.junit.Assert.assertArrayEquals;

import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.junit.Test;

public class IOUtilFunctionsTest {

	@Test
	public void splitCSV_1() {
		String in = "\"1\",\"Prof\",\"B\",19,18,\"Male\",139750";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"1\"", "\"Prof\"", "\"B\"", "19", "18", "\"Male\"", "139750"}, ret);
	}

	@Test
	public void splitCSV_2() {
		String in = "\"1,,,2\",139750";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"1,,,2\"", "139750"}, ret);
	}

	@Test
	public void splitCSV_3() {
		String in = "\"1,,,\"2,139750";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"1,,,\"2", "139750"}, ret);
	}

	@Test
	public void splitCSV_4() {
		String in = "\"1,,\",2,139750";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"1,,\"", "2", "139750"}, ret);
	}

	@Test
	public void splitCSV_5() {
		String in = "\"1,,\"aaaa,2,139750";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"1,,\"aaaa", "2", "139750"}, ret);
	}

	@Test
	public void splitCustom() {
		String in = "0.0,-3.756323061556275,0.0,0.0,9.360046523539289,-2.8007958172584324,-6.233057304650478";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"0.0", "-3.756323061556275", "0.0", "0.0", "9.360046523539289",
			"-2.8007958172584324", "-6.233057304650478"}, ret);

	}

	@Test
	public void splitCustom_2() {
		String in = "aaaaaa \" abb";
		String[] ret = IOUtilFunctions.splitCSV(in, " ");
		assertArrayEquals(new String[] {"aaaaaa", "\"", "abb"}, ret);
	}

	@Test(expected = StringIndexOutOfBoundsException.class)
	public void splitCustom_3() {
		// try{
			String in = "aaaaaa \"\"\" abb";
			String[] ret = IOUtilFunctions.splitCSV(in, " ");
			assertArrayEquals(new String[] {"aaaaaa", "\"\"\"", "abb"}, ret);

		// }
		// catch(Exception e){
			// e.printStackTrace();
			// throw e;
			// fail(e.getMessage());
		// }
	}

	@Test
	public void splitCustom_4() {
		String in = "aaaaaa \"\"\"\" abb";
		String[] ret = IOUtilFunctions.splitCSV(in, " ");
		assertArrayEquals(new String[] {"aaaaaa", "\"\"\"\"", "abb"}, ret);
	}

	@Test
	public void splitCustom_5() {
		String in = "aaaaaa \"\"\"\"";
		String[] ret = IOUtilFunctions.splitCSV(in, " ");
		assertArrayEquals(new String[] {"aaaaaa", "\"\"\"\""}, ret);
	}

	// @Test
	// public void splitCustom_6() {
	// 	String in = "aaaaaa \"\"\"";
	// 	String[] ret = IOUtilFunctions.splitCSV(in, " ");
	// 	assertArrayEquals(new String[] {"aaaaaa", "\""}, ret);
	// }

	@Test
	public void splitCustom_7() {
		String in = "aaaaaa \"";
		String[] ret = IOUtilFunctions.splitCSV(in, " ");
		assertArrayEquals(new String[] {"aaaaaa", "\""}, ret);
	}

	@Test
	public void splitRFC4180Standard_1() {
		String in = "\"aaa\",\"b \r\n\",\"ccc\"";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"aaa\"", "\"b \r\n\"", "\"ccc\""}, ret);
	}

	@Test
	public void splitRFC4180Standard_BreakRule_1() {
		String in = "\"aaa\",\"b\"\"bb\",\"ccc\"";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"aaa\"", "\"b\"\"bb\"", "\"ccc\""}, ret);
	}

	@Test
	public void splitRFC4180Standard_BreakRule_2() {
		String in = "\"aaa\",\"b\"\"bb\"";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"aaa\"", "\"b\"\"bb\""}, ret);
	}

	@Test
	public void splitEmptyMatch_1() {
		String in = "\"aaa\",,,";
		String[] ret = IOUtilFunctions.splitCSV(in, ",");
		assertArrayEquals(new String[] {"\"aaa\"", "", "", ""}, ret);
	}

	@Test
	public void splitEmptyMatch_2() {
		String in = "\"aaa\",,a";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,");
		assertArrayEquals(new String[] {"\"aaa\"", "a"}, ret);
	}

	@Test
	public void split_cache_1() {
		String in = "\"aaa\",,a";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,", new String[2]);
		assertArrayEquals(new String[] {"\"aaa\"", "a"}, ret);
	}

	@Test
	public void split_cache_2() {
		String in = "\"aaa\",,";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,", new String[2]);
		assertArrayEquals(new String[] {"\"aaa\"", ""}, ret);
	}

	@Test
	public void split_cache_3() {
		String in = "";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,", new String[2]);
		assertArrayEquals(new String[] {"", ""}, ret);
	}

	@Test
	public void split_empty_1() {
		String in = "";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,", null);
		assertArrayEquals(new String[] {""}, ret);
	}

	@Test
	public void split_empty_2() {
		String in = "";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,");
		assertArrayEquals(new String[] {""}, ret);
	}

	@Test
	public void split_null() {
		String in = null;
		String[] ret = IOUtilFunctions.splitCSV(in, ",,", null);
		assertArrayEquals(new String[] {""}, ret);
	}

	@Test
	public void split_null_2() {
		String in = null;
		String[] ret = IOUtilFunctions.splitCSV(in, ",,");
		assertArrayEquals(new String[] {""}, ret);
	}

	@Test
	public void splitEmptyMatch_cache_2() {
		String in = "\"aaa\",,a";
		String[] ret = IOUtilFunctions.splitCSV(in, ",,", null);
		assertArrayEquals(new String[] {"\"aaa\"", "a"}, ret);
	}

	@Test
	public void splitCustom_fromRddTest(){
		String in = "aaa,\"\"\",,,b,,\",\"c,c,c\"";

		String[] ret = IOUtilFunctions.splitCSV(in, ",", null);
		assertArrayEquals(new String[] {"aaa", "\"\"\",,,b,,\"", "\"c,c,c\""}, ret);
	}
}
