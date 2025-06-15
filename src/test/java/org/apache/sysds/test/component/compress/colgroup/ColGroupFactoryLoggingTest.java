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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.cost.MemoryCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ColGroupFactoryLoggingTest {
	protected static final Log LOG = LogFactory.getLog(ColGroupFactoryLoggingTest.class.getName());


	@Test 
	public void factoryLoggingTest_accurate(){
		final TestAppender appender = LoggingUtils.overwrite();
		
		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(ColGroupFactory.class).setLevel(Level.TRACE);
			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
			TestUtils.compareMatrices(mb, m2, 0.0);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("wanted:"))
					return;
			}
			fail("Log did not contain wanted message");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(ColGroupFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}


	@Test 
	public void factoryLoggingTest_offestimate(){
		final TestAppender appender = LoggingUtils.overwrite();
		
		try {
			CompressionSettings.printedStatus = false;
			Logger.getLogger(ColGroupFactory.class).setLevel(Level.TRACE);

			MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 30, 1.0, 235);
			mb = TestUtils.floor(mb);

			CompressionSettingsBuilder cs = new CompressionSettingsBuilder().setSamplingRatio(0.02);
			final IColIndex cols = ColIndexFactory.create(mb.getNumColumns());
			final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
			final EstimationFactors f = new EstimationFactors(mb.getNumRows(), mb.getNumRows(), mb.getSparsity());
			es.add(new CompressedSizeInfoColGroup(cols, f, 10, CompressionType.DDC));
			
			CompressedSizeInfo csi = new CompressedSizeInfo(es);
			ACostEstimate ce = new MemoryCostEstimator();
			ColGroupFactory.compressColGroups(mb, csi, cs.create(), ce, 1);
			
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("The estimate cost is significantly off"))
					return;
			}
			// fail("Log did not contain Dictionary sizes");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(ColGroupFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}
}
