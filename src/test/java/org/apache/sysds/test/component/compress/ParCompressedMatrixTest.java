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

package org.apache.sysds.test.component.compress;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class ParCompressedMatrixTest extends AbstractCompressedUnaryTests {

	public ParCompressedMatrixTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettings compressionSettings, MatrixTypology matrixTypology) {
		super(sparType, valType, valRange, compressionSettings, matrixTypology,
			InfrastructureAnalyzer.getLocalParallelism());
	}

	@Override
	public void testUnaryOperators(AggType aggType) {
		AggregateUnaryOperator auop = super.getUnaryOperator(aggType, _k);
		testUnaryOperators(aggType, auop);
	}

}
