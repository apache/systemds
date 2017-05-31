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

package org.apache.sysml.test.integration.gpu;

import org.apache.sysml.test.gpu.AggregateUnaryOpTests;
import org.apache.sysml.test.gpu.BinaryOpTests;
import org.apache.sysml.test.gpu.MatrixMatrixElementWiseOpTests;
import org.apache.sysml.test.gpu.MatrixMultiplicationOpTest;
import org.apache.sysml.test.gpu.NeuralNetworkOpTests;
import org.apache.sysml.test.gpu.ReorgOpTests;
import org.apache.sysml.test.gpu.ScalarMatrixElementwiseOpTests;
import org.apache.sysml.test.gpu.UnaryOpTests;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class) @SuiteClasses({
	BinaryOpTests.class,
    ScalarMatrixElementwiseOpTests.class,
	MatrixMatrixElementWiseOpTests.class,
	ReorgOpTests.class,
	AggregateUnaryOpTests.class,
	UnaryOpTests.class,
	MatrixMultiplicationOpTest.class,
    NeuralNetworkOpTests.class,
})
public class ZPackageSuite {

}
