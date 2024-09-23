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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.fail;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.lib.CLALibCompAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CLALibCompAggLoggingTest {

	protected static final Log LOG = LogFactory.getLog(CLALibCompAggLoggingTest.class.getName());

	@Test
	public void compressedLoggingTest_Trace() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CLALibCompAgg.class).setLevel(Level.TRACE);
			DMLScript.STATISTICS = true;
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			((CompressedMatrixBlock) m2).setOverlapping(true);
			TestUtils.compareMatrices(mb, m2, 0.0);
			
			((CompressedMatrixBlock)m2).clearSoftReferenceToDecompressed();
			TestUtils.compareMatrices(mb.max(10), m2.max(10), 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("decompressed block w/ k"))
					return;
			}
			fail("decompressed block ");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CLALibCompAgg.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}


	@Test
	public void compressedLoggingTest_DEBUG() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CLALibCompAgg.class).setLevel(Level.DEBUG);
			DMLScript.STATISTICS = true;
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			((CompressedMatrixBlock) m2).setOverlapping(true);
			TestUtils.compareMatrices(mb, m2, 0.0);
			
			((CompressedMatrixBlock)m2).clearSoftReferenceToDecompressed();
			TestUtils.compareMatrices(mb.max(10), m2.max(10), 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("decompressed block w/ k"))
					fail("debug should not print decompression block ");
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CLALibCompAgg.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

}
