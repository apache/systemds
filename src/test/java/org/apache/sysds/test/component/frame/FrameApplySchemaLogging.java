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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.fail;

import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.compress.CompressedFrameBlockFactory;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.junit.Test;

public class FrameApplySchemaLogging {

	@Test
	public void testLogging() {
		final TestAppender appender = LoggingUtils.overwrite();
		try {
			Logger.getLogger(FrameLibApplySchema.class).setLevel(Level.TRACE);

			FrameBlock fb = FrameApplySchema.genStringContainingInteger(10, 1);
			ValueType[] schema = new ValueType[] {ValueType.INT32};
			FrameLibApplySchema.applySchema(fb, schema, 3);

			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("Schema Apply Input Size: "))
					return;
			}
			fail("Log did not contain Schema Apply logging");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CompressedFrameBlockFactory.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}
}
