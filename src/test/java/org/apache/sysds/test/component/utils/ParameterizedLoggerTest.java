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
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

import org.apache.commons.logging.Log;
import org.apache.sysds.utils.ParameterizedLogger;
import org.junit.Test;

public class ParameterizedLoggerTest {
	@Test
	public void testGetLogger() {
		assertNotNull(ParameterizedLogger.getLogger(ParameterizedLoggerTest.class));
	}

	@Test
	public void testCurrentTimeMillisIfTraceEnabled() throws Exception {
		Log log = mock(Log.class);
		ParameterizedLogger plog = newLogger(log);

		when(log.isTraceEnabled()).thenReturn(false, true);
		assertEquals(0, plog.currentTimeMillisIfTraceEnabled());
		long ts = plog.currentTimeMillisIfTraceEnabled();
		assertTrue(ts > 0);
	}

	@Test
	public void testIsTraceEnabledAndIsDebugEnabled() throws Exception {
		Log log = mock(Log.class);
		ParameterizedLogger plog = newLogger(log);

		when(log.isTraceEnabled()).thenReturn(true, false);
		when(log.isDebugEnabled()).thenReturn(false, true);
		assertTrue(plog.isTraceEnabled());
		assertEquals(false, plog.isDebugEnabled());

		assertEquals(false, plog.isTraceEnabled());
		assertTrue(plog.isDebugEnabled());
	}

	@Test
	public void testTracePatternAndObject() throws Exception {
		Log log = mock(Log.class);
		ParameterizedLogger plog = newLogger(log);

		when(log.isTraceEnabled()).thenReturn(false, true);
		plog.trace("value {}", 7);
		verify(log, never()).trace(any());

		plog.trace("value {}", 7);
		verify(log).trace(eq("value 7"));

		plog.trace("raw");
		verify(log).trace(eq("raw"));
	}

	@Test
	public void testDebugMethods() throws Exception {
		Log log = mock(Log.class);
		ParameterizedLogger plog = newLogger(log);

		when(log.isDebugEnabled()).thenReturn(false, true);
		plog.debug("x={%5d}", 3);
		verify(log, never()).debug(any());

		plog.debug("x={%5d}", 3);
		verify(log).debug(eq("x=    3"));

		plog.debug("dbg");
		verify(log).debug(eq("dbg"));

		Throwable t = new RuntimeException("boom");
		plog.debug("dbg2", t);
		verify(log).debug(eq("dbg2"), same(t));
	}

	@Test
	public void testInfoAndWarnPatternAndObject() throws Exception {
		Log log = mock(Log.class);
		ParameterizedLogger plog = newLogger(log);

		when(log.isInfoEnabled()).thenReturn(false, true);
		plog.info("i {}", 1);
		verify(log, never()).info(any());

		plog.info("i {}", 1);
		verify(log).info(eq("i 1"));

		plog.info("info");
		verify(log).info(eq("info"));

		when(log.isWarnEnabled()).thenReturn(false, true);
		plog.warn("w {}", 2);
		verify(log, never()).warn(any());

		plog.warn("w {}", 2);
		verify(log).warn(eq("w 2"));

		plog.warn("warn");
		verify(log).warn(eq("warn"));
	}

	@Test
	public void testErrorMethods() throws Exception {
		Log log = mock(Log.class);
		ParameterizedLogger plog = newLogger(log);

		when(log.isErrorEnabled()).thenReturn(false, true);
		plog.error("e {}", 1);
		verify(log, never()).error(any());

		plog.error("e {}", 1);
		verify(log).error(eq("e 1"));

		plog.error("err");
		verify(log).error(eq("err"));

		Throwable t = new IllegalStateException("x");
		plog.error("err2", t);
		verify(log).error(eq("err2"), same(t));
	}

	@Test
	public void testFormatPlainAndStringFormattedPlaceholders() throws Exception {
		assertEquals("a=7 b=      7 c=   3.14000 d=              hi",
			invokeFormat("a={} b={%7d} c={%10.5f} d={%16s}", 7, 7, 3.14, "hi"));
	}

	@Test
	public void testFormatFallbackForInvalidFormatter() throws Exception {
		assertEquals("x=12", invokeFormat("x={%q}", 12));
	}

	@Test
	public void testFormatNonPlaceholderBracesAndExtraArgs() throws Exception {
		assertEquals("prefix {skip} 1 [2]", invokeFormat("prefix {skip} {}", 1, 2));
	}

	private static ParameterizedLogger newLogger(Log log) throws Exception {
		Constructor<ParameterizedLogger> ctor = ParameterizedLogger.class.getDeclaredConstructor(Log.class);
		ctor.setAccessible(true);
		return ctor.newInstance(log);
	}

	private static String invokeFormat(String pattern, Object... args) throws Exception {
		Method method = ParameterizedLogger.class.getDeclaredMethod("format", String.class, Object[].class);
		method.setAccessible(true);
		return (String) method.invoke(null, pattern, args);
	}
}
