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

package org.apache.sysds.test.functions.iogen;

import org.junit.Test;

public class MatrixMultiRowNestedTest extends GenerateReaderMatrixTest {

	private final static String TEST_NAME = "MatrixSingleRowFlatTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	// XML Dataset
	//1. flat object, in-order values
	@Test
	public void test1() {
		sampleRaw = "<a>\n" +
					"<b>1</b>\n" +
					"<c>2</c>\n" +
					"<d>3</d>\n" +
					"</a>\n" +
					"<a>\n" +
					"<b>4</b>\n" +
					"<c>5</c>\n" +
					"<d>6</d>\n" +
					"</a>" +
					"<a>\n" +
					"<b>7</b>\n" +
					"<c>8</c>\n" +
					"<d>9</d>\n" +
					"</a>";
		sampleMatrix = new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		runGenerateReaderTest(false);
	}

	//2. flat object, out-of-order values
	@Test
	public void test2() {
		sampleRaw = "{\"b\":2,\"a\":1,\"e\":5,\"c\":3,\"d\":4}\n" +
					"{\"d\":9,\"b\":7,\"c\":8,\"a\":6,\"e\":10}\n" +
					"{\"d\":14,\"a\":11,\"e\":15,\"b\":12,\"c\":13}";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest(false);
	}
	//3. nested object with unique attribute names
	@Test
	public void test3() {
		sampleRaw = "{\"a\":1,\"b\":{\"c\":2,\"d\":3,\"e\":4},\"f\":5}\n" +
					"{\"a\":6,\"b\":{\"c\":7,\"d\":8,\"e\":9},\"f\":10}\n" +
					"{\"a\":11,\"b\":{\"c\":12,\"d\":13,\"e\":14},\"f\":15}\n";
		sampleMatrix = new double[][] {{1, 2, 5}, {6, 7, 10}, {11, 12, 15}};
		runGenerateReaderTest(false);
	}

	//4. nested object with unique attribute names, out-of-order
	@Test
	public void test4() {
		sampleRaw = "{\"a\":1,\"f\":5,\"b\":{\"c\":2,\"d\":3,\"e\":4}}\n" +
					"{\"a\":6,\"f\":10,\"b\":{\"e\":9,\"c\":7,\"d\":8}}\n" +
					"{\"b\":{\"d\":13,\"c\":12,\"e\":14},\"a\":11,\"f\":15}\n";
		sampleMatrix = new double[][] {{1, 2, 5}, {6, 7, 10}, {11, 12, 15}};
		runGenerateReaderTest(false);
	}

	//5. nested object with repeated attribute names, out-of-order
	@Test
	public void test5() {
		sampleRaw = "{\"a\":1,\"b\":{\"a\":2,\"b\":3,\"f\":4},\"f\":5}\n" +
					"{\"a\":6,\"b\":{\"a\":7,\"b\":8,\"f\":9},\"f\":10}\n" +
					"{\"a\":11,\"b\":{\"a\":12,\"b\":13,\"f\":14},\"f\":15}";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest(false);
	}

	// XML
	//6. nested object with unique attribute names, in-order
	// single type of object, "article" is an object
	@Test
	public void test6() {
		sampleRaw = "<article><a>1</a><b>2</b><c>3</c><d>4</e><f>5</f></article>\n" +
					"<article><a>6</a><b>7</b><c>8</c><d>9</e><f>10</f></article>\n" +
					"<article><a>11</a><b>12</b><c>13</c><d>14</e><f>15</f></article>";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest(false);
	}

	//6. nested object with unique attribute names, in-order
	// multi types of object, "article", "book", and "homepage" are the object types
	@Test
	public void test7() {
		sampleRaw = "<article><a>1</a><b>2</b><c>3</c><d>4</e><f>5</f></article>\n" +
					"<book><a>6</a><b>7</b><c>8</c><d>9</e><f>10</f></book>\n" +
					"<homepage><a>11</a><b>12</b><c>13</c><d>14</e><f>15</f></homepage>";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest(false);
	}

	//7. nested object with unique attribute names, in-order
	// multi types of object, "article", "book", and "homepage" are the object types
	@Test
	public void test8() {
		sampleRaw = "<article><a>1</a><b>2</b><c><year>2022</year><title>GIO</title></c><d>4</e><f>5</f></article>\n" +
					"<book><a>6</a><b>7</b><c><year>1980</year><title>DB</title></c><d>9</e><f>10</f></book>\n" +
					"<homepage><a>11</a><b>12</b><c><year>2012</year><title>CEP</title></c><d>14</e><f>15</f></homepage>\n";
		sampleMatrix = new double[][] {{1, 2022}, {6, 1980}, {11, 2012}};
		runGenerateReaderTest(false);
	}

	@Test
	public void test9() {
		sampleRaw = "#index 1\n" +
			"#* 12\n" +
			"#@ 13;14;15\n" +
			"#o 16;17;18\n" +
			"#t 19\n" +
			"#c 110\n" +
			"#% 111\n" +
			"#% 112\n" +
			"\n" +
			"#index 2\n" +
			"#* 22\n" +
			"#@ 23;24;25\n" +
			"#o 26;27;28\n" +
			"#t 29\n" +
			"#c 210\n" +
			"#% 211\n" +
			"#% 212\n" +
			"\n" +
			"\n" +
			"#index 3\n" +
			"#* 32\n" +
			"#@ 33;34;35\n" +
			"#o 36;37;38\n" +
			"#t 39\n" +
			"#c 310\n" +
			"#% 311\n" +
			"#% 500\n"+
			"\n" +
			"#index 4\n" +
			"#* 42\n" +
			"#@ 43;44;45\n" +
			"#o 46;47;48\n" +
			"#t 49\n" +
			"#c 410\n" +
			"#% 411\n" +
			"#% 600";

		sampleMatrix = new double[][] {{1,12,13,14,15},{2,22,23,24,25},{3,32,33,34,35}, {4,42,43,44,45}};
		runGenerateReaderTest(false);
	}
}
