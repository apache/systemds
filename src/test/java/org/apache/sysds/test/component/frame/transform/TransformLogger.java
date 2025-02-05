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

package org.apache.sysds.test.component.frame.transform;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.CompressedEncode;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class TransformLogger {
	protected static final Log LOG = LogFactory.getLog(TransformLogger.class.getName());

	private final FrameBlock data;

	public TransformLogger() {
		try {

			data = TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT4}, 231);
			data.setSchema(new ValueType[] {ValueType.INT32});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
			throw e;
		}
	}

	@Test
	public void testDummyCode() {
		test("{dummycode:[C1]}");
	}

	public void test(String spec) {
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CompressedEncode.class).setLevel(Level.DEBUG);
			Logger.getLogger(DependencyThreadPool.class).setLevel(Level.DEBUG);
			Logger.getLogger(DependencyTask.class).setLevel(Level.DEBUG);

			FrameBlock meta = null;
			MultiColumnEncoder encoderNormal = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);
			encoderNormal.encode(data, 10);

			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);

			boolean containsMessage = false;
			for(LoggingEvent l : log) {
				containsMessage |= l.getMessage().toString().contains("EXPlAIN (TASK-GRAPH):");
			}

			assertTrue(containsMessage);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			LoggingUtils.reinsert(appender);
		}

	}

}
