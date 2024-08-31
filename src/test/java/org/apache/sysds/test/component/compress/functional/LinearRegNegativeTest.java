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

package org.apache.sysds.test.component.compress.functional;

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.sysds.runtime.compress.colgroup.functional.LinearRegression;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class LinearRegNegativeTest {

	@Test(expected = Exception.class)
	public void invalidRows() {
		LinearRegression.regressMatrixBlock(new MatrixBlock(-1, -1, 132), null, false);
	}

	@Test(expected = Exception.class)
	public void invalidCols() {

		IColIndex spy = spy(ColIndexFactory.create(10));
		when(spy.size()).thenReturn(-1);
		
		LinearRegression.regressMatrixBlock(new MatrixBlock(10, 10, 132), spy, false);
	}


}
