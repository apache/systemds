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

package org.apache.sysds.test.component.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.sysds.utils.ParameterizedLogger;
import org.junit.Test;

public class ParameterizedLoggerTest {

	@Test
	public void testFormat() {
		assertEquals("null", ParameterizedLogger.format(null));
		assertEquals("pattern", ParameterizedLogger.format("pattern"));
		assertEquals("a=7 b=      7 c=   3.14000 d=              hi",
			ParameterizedLogger.format("a={} b={%7d} c={%10.5f} d={%16s}", 7, 7, 3.14, "hi"));
		assertEquals("x=12", ParameterizedLogger.format("x={%q}", 12));
		assertEquals("prefix {skip} 1 [2]", ParameterizedLogger.format("prefix {skip} {}", 1, 2));
		assertEquals("a=1 b={}", ParameterizedLogger.format("a={} b={}", 1));
		assertEquals("v=null", ParameterizedLogger.format("v={}", new Object[] {null}));
	}

	@Test
	public void testDispatchAndGuards() {
		RecordingLog log = new RecordingLog();
		ParameterizedLogger plog = new ParameterizedLogger(log);

		log.debugEnabled = false;
		plog.debug("x={%5d}", 3);
		assertNull(log.lastMessage);

		log.debugEnabled = true;
		plog.debug("x={%5d}", 3);
		assertEquals("x=    3", log.lastMessage);

		plog.debug("plain");
		assertEquals("plain", log.lastMessage);

		Throwable t = new RuntimeException("boom");
		plog.debug("dbg2", t);
		assertEquals("dbg2", log.lastMessage);
		assertSame(t, log.lastThrowable);
	}

	@Test
	public void testFactoryAndLevelPassthrough() {
		assertNotNull(ParameterizedLogger.getLogger(ParameterizedLoggerTest.class));

		RecordingLog log = new RecordingLog();
		ParameterizedLogger plog = new ParameterizedLogger(log);

		log.traceEnabled = false;
		assertEquals(0, plog.currentTimeMillisIfTraceEnabled());
		assertEquals(false, plog.isTraceEnabled());

		log.traceEnabled = true;
		assertTrue(plog.currentTimeMillisIfTraceEnabled() > 0);
		assertTrue(plog.isTraceEnabled());

		log.debugEnabled = true;
		assertTrue(plog.isDebugEnabled());

		log.infoEnabled = true;
		assertTrue(plog.isInfoEnabled());
	}

	@Test
	public void testLevelGuards() {
		RecordingLog log = new RecordingLog();
		ParameterizedLogger plog = new ParameterizedLogger(log);

		log.traceEnabled = false;
		plog.trace("t={}", 1);
		assertNull(log.lastMessage);
		log.traceEnabled = true;
		plog.trace("t={}", 1);
		assertEquals("t=1", log.lastMessage);

		log.lastMessage = null;
		log.infoEnabled = false;
		plog.info("i={}", 2);
		assertNull(log.lastMessage);
		log.infoEnabled = true;
		plog.info("i={}", 2);
		assertEquals("i=2", log.lastMessage);

		log.lastMessage = null;
		log.warnEnabled = false;
		plog.warn("w={}", 3);
		assertNull(log.lastMessage);
		log.warnEnabled = true;
		plog.warn("w={}", 3);
		assertEquals("w=3", log.lastMessage);

		log.lastMessage = null;
		log.errorEnabled = false;
		plog.error("e={}", 4);
		assertNull(log.lastMessage);
		log.errorEnabled = true;
		plog.error("e={}", 4);
		assertEquals("e=4", log.lastMessage);
	}

	private static final class RecordingLog implements Log {
		boolean traceEnabled;
		boolean debugEnabled;
		boolean infoEnabled = true;
		boolean warnEnabled = true;
		boolean errorEnabled = true;
		Object lastMessage;
		Throwable lastThrowable;

		private void record(Object message, Throwable t) {
			lastMessage = message;
			lastThrowable = t;
		}

		@Override
		public boolean isTraceEnabled() {
			return traceEnabled;
		}

		@Override
		public boolean isDebugEnabled() {
			return debugEnabled;
		}

		@Override
		public boolean isInfoEnabled() {
			return infoEnabled;
		}

		@Override
		public boolean isWarnEnabled() {
			return warnEnabled;
		}

		@Override
		public boolean isErrorEnabled() {
			return errorEnabled;
		}

		@Override
		public boolean isFatalEnabled() {
			return true;
		}

		@Override
		public void trace(Object message) {
			record(message, null);
		}

		@Override
		public void trace(Object message, Throwable t) {
			record(message, t);
		}

		@Override
		public void debug(Object message) {
			record(message, null);
		}

		@Override
		public void debug(Object message, Throwable t) {
			record(message, t);
		}

		@Override
		public void info(Object message) {
			record(message, null);
		}

		@Override
		public void info(Object message, Throwable t) {
			record(message, t);
		}

		@Override
		public void warn(Object message) {
			record(message, null);
		}

		@Override
		public void warn(Object message, Throwable t) {
			record(message, t);
		}

		@Override
		public void error(Object message) {
			record(message, null);
		}

		@Override
		public void error(Object message, Throwable t) {
			record(message, t);
		}

		@Override
		public void fatal(Object message) {
		}

		@Override
		public void fatal(Object message, Throwable t) {
		}
	}
}
