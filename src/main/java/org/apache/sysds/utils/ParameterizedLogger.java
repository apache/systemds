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

package org.apache.sysds.utils;

import java.util.IllegalFormatException;
import java.util.Locale;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Logger adapter for commons-logging with lightweight '{}' parameter formatting.
 */
public class ParameterizedLogger {
	private final Log _log;

	private ParameterizedLogger(Log log) {
		_log = log;
	}

	public static ParameterizedLogger getLogger(Class<?> cls) {
		return new ParameterizedLogger(LogFactory.getLog(cls.getName()));
	}

	public long currentTimeMillisIfTraceEnabled() {
		return _log.isTraceEnabled() ? System.currentTimeMillis() : 0;
	}

	public boolean isTraceEnabled() {
		return _log.isTraceEnabled();
	}

	public boolean isDebugEnabled() {
		return _log.isDebugEnabled();
	}
	
	
	public void trace(String pattern, Object... args) {
		if(_log.isTraceEnabled())
			_log.trace(format(pattern, args));
	}

	public void trace(Object message) {
		_log.trace(message);
	}

	public void debug(String pattern, Object... args) {
		if(_log.isDebugEnabled())
			_log.debug(format(pattern, args));
	}

	public void debug(Object message) {
		_log.debug(message);
	}

	public void debug(Object message, Throwable t) {
		_log.debug(message, t);
	}

	public void info(String pattern, Object... args) {
		if(_log.isInfoEnabled())
			_log.info(format(pattern, args));
	}

	public void info(Object message) {
		_log.info(message);
	}

	public void warn(String pattern, Object... args) {
		if(_log.isWarnEnabled())
			_log.warn(format(pattern, args));
	}

	public void warn(Object message) {
		_log.warn(message);
	}


	public void error(String pattern, Object... args) {
		if(_log.isErrorEnabled())
			_log.error(format(pattern, args));
	}

	public void error(Object message) {
		_log.error(message);
	}

	public void error(Object message, Throwable t) {
		_log.error(message, t);
	}

	private static String format(String pattern, Object... args) {
		if(pattern == null)
			return "null";
		if(args == null || args.length == 0)
			return pattern;

		StringBuilder sb = new StringBuilder(pattern.length() + args.length * 16);
		int argIx = 0;
		int start = 0;
		int search = 0;
		int open;
		while((open = pattern.indexOf('{', search)) >= 0 && argIx < args.length) {
			int close = pattern.indexOf('}', open + 1);
			if(close < 0)
				break;

			boolean defaultPlaceholder = close == open + 1;
			boolean stringFormatPlaceholder = !defaultPlaceholder && pattern.charAt(open + 1) == '%';
			if(!defaultPlaceholder && !stringFormatPlaceholder) {
				search = open + 1;
				continue;
			}

			sb.append(pattern, start, open);
			if(defaultPlaceholder)
				sb.append(String.valueOf(args[argIx++]));
			else
				sb.append(formatArgument(pattern.substring(open + 1, close), args[argIx++]));
			start = close + 1;
			search = start;
		}
		sb.append(pattern, start, pattern.length());

		while(argIx < args.length) {
			sb.append(" [");
			sb.append(String.valueOf(args[argIx++]));
			sb.append(']');
		}

		return sb.toString();
	}

	private static String formatArgument(String format, Object arg) {
		try {
			return String.format(Locale.ROOT, format, arg);
		}
		catch(IllegalFormatException ex) {
			return String.valueOf(arg);
		}
	}
}
