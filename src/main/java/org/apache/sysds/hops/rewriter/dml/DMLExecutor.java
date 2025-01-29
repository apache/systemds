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

package org.apache.sysds.hops.rewriter.dml;

import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public class DMLExecutor {
	private static PrintStream origPrintStream = System.out;
	private static PrintStream origErrPrintStream = System.out;

	public static boolean APPLY_INJECTED_REWRITES = false;
	public static Function<Hop, Hop> REWRITE_FUNCTION = null;

	private static List<String> lastErr;

	public static void executeCode(String code, boolean intercept, String... additionalArgs) {
		executeCode(code, intercept ? s -> {} : null, additionalArgs);
	}

	// Returns if true if the run was successful without any errors
	public static boolean executeCode(String code, Consumer<String> consoleInterceptor, String... additionalArgs) {
		return executeCode(code, consoleInterceptor, null, additionalArgs);
	}

	// This cannot run in parallel
	public static synchronized boolean executeCode(String code, Consumer<String> consoleInterceptor, Function<Hop, Hop> injectedRewriteClass, String... additionalArgs) {
		lastErr = new ArrayList<>();
		boolean exceptionOccurred = false;

		try {
			if (consoleInterceptor != null)
				System.setOut(new PrintStream(new CustomOutputStream(System.out, consoleInterceptor)));

			System.setErr(new PrintStream(new CustomOutputStream(System.err, lastErr::add)));

			String[] args = new String[additionalArgs.length + 2];

			for (int i = 0; i < additionalArgs.length; i++)
				args[i] = additionalArgs[i];

			args[additionalArgs.length] = "-s";
			args[additionalArgs.length + 1] = code;

			if (injectedRewriteClass != null) {
				APPLY_INJECTED_REWRITES = true;
				REWRITE_FUNCTION = injectedRewriteClass;
			}

			// To allow the discovery of sum((a*A)*B) which would usually be converted to n*
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = false;

			DMLScript.executeScript(args);

		} catch (Exception e) {
			e.printStackTrace();
			exceptionOccurred = true;
		}

		APPLY_INJECTED_REWRITES = false;
		REWRITE_FUNCTION = null;

		if (consoleInterceptor != null)
			System.setOut(origPrintStream);

		System.setErr(origErrPrintStream);

		return !exceptionOccurred && lastErr.isEmpty();
	}

	public static List<String> getLastErr() {
		return lastErr;
	}

	// Bypasses the interceptor
	public static void println(Object o) {
		origPrintStream.println(o);
	}

	private static class CustomOutputStream extends OutputStream {
		private PrintStream ps;
		private StringBuilder buffer = new StringBuilder();
		private Consumer<String> lineHandler;

		public CustomOutputStream(PrintStream actualPrintStream, Consumer<String> lineHandler) {
			this.ps = actualPrintStream;
			this.lineHandler = lineHandler;
		}

		@Override
		public void write(int b) {
			char c = (char) b;
			if (c == '\n') {
				lineHandler.accept(buffer.toString());
				buffer.setLength(0); // Clear the buffer after handling the line
			} else {
				buffer.append(c); // Accumulate characters until newline
			}
		}

		@Override
		public void write(byte[] b, int off, int len) {
			for (int i = off; i < off + len; i++) {
				write(b[i]);
			}
		}
	}
}
