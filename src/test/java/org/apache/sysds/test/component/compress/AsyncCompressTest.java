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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class AsyncCompressTest {
	protected static final Log LOG = LogFactory.getLog(AsyncCompressTest.class.getName());

	@Test
	public void empty() {
		assertTrue(runTest(new MatrixBlock(1000, 30, 0.0)));
	}

	@Test
	public void notCompressable() {
		assertFalse(runTest(TestUtils.generateTestMatrixBlock(100, 100, 0, 1.0, 1.0, 13)));
	}

	public boolean runTest(MatrixBlock mb) {
		try {
			MatrixCharacteristics matrixCharacteristics = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(),
				-1, 0);
			MetaDataFormat metaDataFormat = new MetaDataFormat(matrixCharacteristics, FileFormat.TEXT);

			MatrixObject mbo = new MatrixObject(ValueType.FP64, "/dev/null", metaDataFormat, mb);
			LocalVariableMap vars = new LocalVariableMap();
			ExecutionContext ec = new ExecutionContext(vars);
			ec.setVariable("mb1", mbo);

			CompressedMatrixBlockFactory.compressAsync(ec, "mb1");

			for(int i = 0; i < 5; i++) {
				Thread.sleep(i * 100);
				MatrixBlock m = mbo.acquireReadAndRelease();
				if(m instanceof CompressedMatrixBlock)
					return true;
			}
			return false;
		}

		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
			throw new DMLRuntimeException("failed test", e);
		}

	}
}
