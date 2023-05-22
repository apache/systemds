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

package org.apache.sysds.test;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Appender;
import org.apache.log4j.AppenderSkeleton;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;

/**
 * Logging utils, to enable catching the Log output in tests.
 * 
 * To use simply start with overwrite and when the test is done remember to use reinsert.
 */
public final class LoggingUtils {

	private static Appender consoleLogger = null;

	private LoggingUtils() {
		// empty constructor
	}

	/**
	 * Overwrite the console logger to not write out to STD error.
	 * 
	 * @return A TestAppender that collects the logging outputs.
	 */
	public static TestAppender overwrite() {
		if(consoleLogger == null)
			findConsoleLogger();
		final TestAppender appender = new TestAppender();
		final Logger logger = Logger.getRootLogger();
		logger.removeAppender(consoleLogger);
		logger.addAppender(appender);

		return appender;
	}

	/**
	 * Reinstate the console logger, usually done after the test.
	 * 
	 * @param appender The appender that was used to collect the Logging outputs
	 * @return The List of logging statements done while the Test Appender was in use
	 */
	public static List<LoggingEvent> reinsert(TestAppender appender) {
		final Logger logger = Logger.getRootLogger();
		logger.removeAppender(appender);
		logger.addAppender(consoleLogger);
		return appender.getLog();
	}

	private static void findConsoleLogger() {
		final Logger logger = Logger.getRootLogger();
		consoleLogger = (Appender) logger.getAllAppenders().nextElement();
	}

	/**
	 * A Test Appender that collects the Logging calls into a list.
	 * 
	 * To be used in connection with LoggingUtils.
	 */
	public static class TestAppender extends AppenderSkeleton {
		private final List<LoggingEvent> log = new ArrayList<LoggingEvent>();

		private TestAppender() {
			// empty constructor
		}

		@Override
		public boolean requiresLayout() {
			return false;
		}

		@Override
		protected void append(final LoggingEvent loggingEvent) {
			log.add(loggingEvent);
		}

		@Override
		public void close() {
		}

		private List<LoggingEvent> getLog() {
			return new ArrayList<LoggingEvent>(log);
		}
	}
}
