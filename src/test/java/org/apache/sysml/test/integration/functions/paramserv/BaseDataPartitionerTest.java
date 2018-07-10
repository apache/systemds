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

package org.apache.sysml.test.integration.functions.paramserv;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;

public abstract class BaseDataPartitionerTest {

	protected static final int ROW_SIZE = 4000;
	protected static final int COL_SIZE = 2000;
	protected static final int WORKER_NUM = 4;

	protected MatrixBlock[] generateData() {
		double[][] df = new double[BaseDataPartitionerTest.ROW_SIZE][BaseDataPartitionerTest.COL_SIZE];
		for (int i = 0; i < BaseDataPartitionerTest.ROW_SIZE; i++) {
			for (int j = 0; j < BaseDataPartitionerTest.COL_SIZE; j++) {
				df[i][j] = i * BaseDataPartitionerTest.COL_SIZE + j;
			}
		}
		double[] dl = new double[BaseDataPartitionerTest.ROW_SIZE];
		for (int i = 0; i < BaseDataPartitionerTest.ROW_SIZE; i++) {
			dl[i] = i;
		}
		MatrixBlock fmb = DataConverter.convertToMatrixBlock(df);
		MatrixBlock lmb = DataConverter.convertToMatrixBlock(dl, true);
		return new MatrixBlock[] { fmb, lmb };
	}
}
