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

import static org.junit.Assert.fail;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

public class CompressedLoggingTests {
	protected static final Log 	LOG = LogFactory.getLog(CompressedLoggingTests.class.getName());
	
	@Test
	public void compressedLoggingTest_Trace() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.TRACE);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("compressed colGroup dictionary sizes"))
					return;
			}
			fail("Log did not contain Dictionary sizes");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_WorkloadCost() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
			InstructionTypeCounter inst = new InstructionTypeCounter(10, 0, 0, 0, 0, 0, 0, 0, false);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb, inst).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("--actual cost:"))
					return;
			}
			fail("Log did not contain Dictionary sizes");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {

			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_ManyColumns() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(2000, 1001, 1, 65, 0.5, 235);
			mb = TestUtils.round(mb);
			InstructionTypeCounter inst = new InstructionTypeCounter(0, 0, 0, 0, 0, 1000, 0, 0, false);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb, inst).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("--CoCoded produce many columns but the first"))
					return;
			}
			fail("Log did not say Cocode many columns");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_failedCompression() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);

			MatrixBlock mb = TestUtils.generateTestMatrixBlock(5, 5, 0.2, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Aborting before co-code"))
					return;
			}
			fail("Log did not contain abort");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_failedCompression_afterCocode() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);

			MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 50, 1, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Aborting after co-code"))
					return;
			}
			fail("Log did not contain abort");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_WorkloadCostFail() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 50, 1, 1, 0.5, 235);
			InstructionTypeCounter inst = new InstructionTypeCounter(10, 0, 0, 0, 0, 0, 0, 0, false);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb, inst).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Aborting after co-code"))
					return;
			}
			fail("Log did not contain abort");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_WorkloadCostFail_2() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 50, 1, 1, 0.5, 235);
			InstructionTypeCounter inst = new InstructionTypeCounter(0, 10, 0, 0, 0, 0, 0, 0, true);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb, inst).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Aborting before co-code"))
					return;
			}
			fail("Log did not contain abort");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_SparkSettings() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 2, 1, 1, 0.5, 235);
			CompressionSettingsBuilder sb = new CompressionSettingsBuilder();
			sb.setIsInSparkInstruction();
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb, sb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Compressed Size"))
					return;
			}
			fail("Log did not contain Compressed Size");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_TraceBigGroup() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.TRACE);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(10000, 1, 1, 128, 0.5, 235);
			mb = TestUtils.round(mb);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("--colGroups type"))
					return;
			}
			fail("Log did not contain colgroups type");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_TraceBigGroupConst() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.TRACE);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 1000, 1, 1, 1.0, 235);
			mb = TestUtils.round(mb);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("--colGroups type"))
					return;
			}
			fail("Log did not contain colgroups type");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTestEmpty() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.TRACE);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 1000, 1, 1, 0.0, 235);
			mb = TestUtils.round(mb);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Empty input to compress"))
					return;
			}
			fail("Log did not contain Empty");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_recompress() {
		TestAppender appender = null;
		
		try {
			CompressionSettings.printedStatus = false;
			appender = LoggingUtils.overwrite();
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 3, 1, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			CompressedMatrixBlockFactory.compress(m2).getLeft();
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
		
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Recompressing"))
					return;
			}
			fail("Log did not contain Recompressing");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	@Ignore
	public void compressedLoggingTest_AbortEnd() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.DEBUG);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(400, 600, 1, 1024, 0.18, 235);
			mb = TestUtils.round(mb);
			final int ss = 50;
			CompressionSettingsBuilder sb = new CompressionSettingsBuilder();
			sb.setMaxSampleSize(ss);
			sb.setMinimumSampleSize(ss);
			CompressedMatrixBlockFactory.compress(mb, sb).getLeft();
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Abort block compression"))
					return;
			}
			fail("Log did not contain abort block compression");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressionSettings() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(CompressionSettings.class).setLevel(Level.DEBUG);
			new CompressionSettingsBuilder().create();
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("CompressionSettings"))
					return;
			}
			fail("failed to get Compression setting to log");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressionSettings.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressionSettingsEstimationType() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			CompressionSettings.printedStatus = false;

			Logger.getLogger(CompressionSettings.class).setLevel(Level.DEBUG);
			new CompressionSettingsBuilder().setSamplingRatio(0.1).create();
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Estimation Type"))
					return;
			}
			fail("failed to get estimation type");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressionSettings.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressionSettingsFull() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CompressionSettings.class).setLevel(Level.DEBUG);
			new CompressionSettingsBuilder().setSamplingRatio(1.1).create();
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Estimation Type"))
					fail("Contained estimationType");
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressionSettings.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

	@Test
	public void compressedLoggingTest_NNzNotSet() {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
			mb.setNonZeros(-1);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Recomputing non-zeros"))
					return;
			}
			fail("NonZeros not set warning not printed");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedMatrixBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}
}
