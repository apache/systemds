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

public class MatrixSingleRowFlatTest extends GenerateReaderMatrixTest {

	private final static String TEST_NAME = "MatrixSingleRowFlatTest";

	@Override protected String getTestName() {
		return TEST_NAME;
	}

	// CSV Dataset
	// 1. matrix and dataset are dense and "," is delim
	@Test public void test1() {
		sampleRaw = "1,2,3,4,5\n" + "6,7,8,9,10\n" + "11,12,13,14,15";
		sampleMatrix = new double[][] {{1, 2}, {6, 7}, {11, 12}};
		runGenerateReaderTest(false);
	}

	// 2. matrix and dataset are dense and ",a" is delim
	@Test public void test2() {
		sampleRaw = "1,a2,a3,a4,a5\n" + "6,a7,a8,a9,a10\n" + "11,a12,a13,a14,a15";
		sampleMatrix = new double[][] {{1, 5}, {6, 10}, {11, 15}};
		runGenerateReaderTest(false);
	}

	//3. matrix and dataset are dense and ",," is delim
	@Test public void test3() {
		sampleRaw = "1,,2,,3,,4,,5\n" + "6,,7,,8,,9,,10\n" + "11,,12,,13,,14,,15";
		sampleMatrix = new double[][] {{1, 3, 5}, {6, 8, 10}, {11, 13, 15}};
		runGenerateReaderTest(false);
	}

	//4. matrix and dataset contain empty/0 values and "," is delim
	@Test public void test4() {
		sampleRaw = "1,2,,4,5\n" + ",7,8,9,10\n" + "11,12,,,\n" + "13,14,,,16";
		sampleMatrix = new double[][] {{1, 2, 5}, {0, 7, 10}, {11, 12, 0}, {13, 14, 16}};
		runGenerateReaderTest(false);
	}

	// LibSVM
	//5. LibSVM with in-order col indexes and numeric col indexes
	@Test public void test5() {
		sampleRaw = "+1 1:10 2:20 3:30\n" + "-1 4:40 5:50 6:60\n" + "+1 1:101 2:201 \n" + "-1 6:601 \n" + "-1 5:501\n" + "+1 3:301";
		sampleMatrix = new double[][] {{1, 10, 20, 30, 0, 0, 0}, {-1, 0, 0, 0, 40, 50, 60}, {1, 101, 201, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0, 601},
			{-1, 0, 0, 0, 0, 501, 0}, {1, 0, 0, 301, 0, 0, 0}};
		runGenerateReaderTest(false);
	}

	//6. LibSVM with out-of-order col indexes and numeric col indexes
	@Test public void test6() {
		sampleRaw = "+1 3:30 1:10 2:20\n" + "-1 5:50 6:60 4:40\n" + "+1 1:101 2:201 \n" + "-1 6:601 \n" + "-1 5:501\n" + "+1 3:301";
		sampleMatrix = new double[][] {{1, 10, 20, 30, 0, 0, 0}, {-1, 0, 0, 0, 40, 50, 60}, {1, 101, 201, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0, 601},
			{-1, 0, 0, 0, 0, 501, 0}, {1, 0, 0, 301, 0, 0, 0}};
		runGenerateReaderTest(false);
	}

	//7. Special LibSVM with in-order col indexes and none-numeric col indexes
	// a -> 1, b->2, c->3, d->4, e->5, f->6
	@Test public void test7() {
		sampleRaw = "+1 a:10 b:20 c:30\n" + "-1 d:40 e:50 f:60\n" + "+1 a:101 b:201 \n" + "-1 f:601 \n" + "-1 e:501\n" + "+1 c:301";
		sampleMatrix = new double[][] {{1, 10, 20, 30, 0, 0, 0}, {-1, 0, 0, 0, 40, 50, 60}, {1, 101, 201, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0, 601},
			{-1, 0, 0, 0, 0, 501, 0}, {1, 0, 0, 301, 0, 0, 0}};
		runGenerateReaderTest(false);
	}

	//8. Special LibSVM with out-of-order col indexes and none-numeric col indexes
	// a -> 1, b->2, c->3, d->4, e->5, f->6
	@Test public void test8() {
		sampleRaw = "+1 c:30 a:10 b:20\n" + "-1 e:50 f:60 d:40\n" + "+1 a:101 b:201 \n" + "-1 f:601 \n" + "-1 e:501\n" + "+1 c:301";
		sampleMatrix = new double[][] {{1, 10, 20, 30, 0, 0, 0}, {-1, 0, 0, 0, 40, 50, 60}, {1, 101, 201, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0, 601},
			{-1, 0, 0, 0, 0, 501, 0}, {1, 0, 0, 301, 0, 0, 0}};
		runGenerateReaderTest(false);
	}

	// MatrixMarket(MM)
	//9. MM with inorder dataset, (RowIndex,Col Index,Value). Row & Col begin index: (1,1)
	@Test public void test9() {
		sampleRaw = "1,1,10\n" + "1,2,20\n" + "1,3,30\n" + "1,5,50\n" + "2,1,101\n" + "2,2,201\n" + "4,1,104\n" + "4,5,504\n" + "5,3,305";
		sampleMatrix = new double[][] {{10, 20, 30}, {101, 201, 0}, {0, 0, 0}, {104, 0, 0}, {0, 0, 305}};
		runGenerateReaderTest(false);
	}

	//10. MM with inorder dataset, (RowIndex,Col Index,Value). Row begin index: Row & Col begin index: (0,1)
	@Test public void test10() {
		sampleRaw = "0,1,10\n" + "0,2,20\n" + "0,3,30\n" + "0,5,50\n" + "1,1,101\n" + "1,2,201\n" + "3,1,104\n" + "3,5,504\n" + "4,3,305";
		sampleMatrix = new double[][] {{10, 20, 30}, {101, 201, 0}, {0, 0, 0}, {104, 0, 0}, {0, 0, 305}};
		runGenerateReaderTest(false);
	}

	//11. MM with inorder dataset, (RowIndex,Col Index,Value). Row & Col begin index: (1,0)
	@Test public void test11() {
		sampleRaw = "1,0,10\n" + "1,1,20\n" + "1,2,30\n" + "1,4,50\n" + "2,0,101\n" + "2,1,201\n" + "4,0,104\n" + "4,4,504\n" + "5,2,305";
		sampleMatrix = new double[][] {{10, 20, 30}, {101, 201, 0}, {0, 0, 0}, {104, 0, 0}, {0, 0, 305}};
		runGenerateReaderTest(false);
	}

	//12. MM with inorder dataset, (RowIndex,Col Index,Value). Row begin index: Row & Col begin index: (0,0)
	@Test public void test12() {
		sampleRaw = "0,0,10\n" + "0,1,20\n" + "0,2,30\n" + "0,4,50\n" + "1,0,101\n" + "1,1,201\n" + "2,0,104\n" + "2,4,504\n" + "3,2,305";
		sampleMatrix = new double[][] {{10, 20, 30}, {101, 201, 0}, {104, 0, 0}, {0, 0, 305}};
		runGenerateReaderTest(false);
	}

	//13. MM with out-of-order dataset, (RowIndex,Col Index,Value). Row & Col begin index: (1,1)
	@Test public void test13() {
		sampleRaw = "4,5,504\n" + "1,2,20\n" + "1,1,10\n" + "2,1,101\n" + "1,3,30\n" + "1,5,50\n" + "2,2,201\n" + "4,1,104\n" + "5,3,305";
		sampleMatrix = new double[][] {{10, 20, 30}, {101, 201, 0}, {0, 0, 0}, {104, 0, 0}, {0, 0, 305}};
		runGenerateReaderTest(false);
	}

	//14. MM with out-of-order dataset, (ColIndex,Row Index, Value). Row & Col begin index: (1,1)
	@Test public void test14() {
		sampleRaw = "5,4,504\n" + "2,1,20\n" + "1,1,10\n" + "1,2,101\n" + "3,1,30\n" + "5,1,50\n" + "2,2,201\n" + "1,4,104\n" + "3,5,305\n" + "2,4,204";
		sampleMatrix = new double[][] {{10, 20, 30}, {101, 201, 0}, {0, 0, 0}, {104, 204, 0}, {0, 0, 305}};
		runGenerateReaderTest(false);
	}

	@Test public void test15() {
		sampleRaw = "1,2,3,4\n" + "5,6,7,8\n" + "9,10,11,12\n" + "13,14,15,16";
		sampleMatrix = new double[][] {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
		runGenerateReaderTest(false);
	}

	@Test public void test16() {
		sampleRaw = "1,2,3,0\n" + "5,0,7,8\n" + "9,0,0,12\n" + "13,14,0,0";
		sampleMatrix = new double[][] {{1, 2, 3, 0}, {5, 0, 7, 8}, {9, 0, 0, 12}, {13, 14, 0, 0}};
		runGenerateReaderTest(false);
	}

	@Test public void test17() {
		sampleRaw = "0:10 1:20 2:30\n" + "3:40 4:50\n" + "0:60 1:70 2:80\n" + "3:90 4:100";
		sampleMatrix = new double[][] {{10, 20, 30, 0, 0}, {0, 0, 0, 40, 50}, {60, 70, 80, 0, 0}, {0, 0, 0, 90, 100}};
		runGenerateReaderTest(false);
	}

	@Test public void test18() {
		sampleRaw = "1,1,10\n" + "1,2,20\n" + "1,3,30\n" + "1,4,40\n" + "2,2,20\n" + "2,3,30\n" + "2,4,40\n" + "3,3,30\n" + "3,4,40\n" + "4,4,40\n";
		sampleMatrix = new double[][] {{10, 20, 30, 40}, {0, 20, 30, 40}, {0, 0, 30, 40}, {0, 0, 0, 40}};
		runGenerateReaderTest(false);
	}
}
