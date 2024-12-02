package org.apache.sysds.hops.rewriter.dml;

import org.apache.sysds.api.DMLScript;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.function.Consumer;

public class DMLExecutor {
	private static PrintStream origPrintStream = System.out;

	public static synchronized void executeCode(String code, boolean intercept, String... additionalArgs) {
		executeCode(code, intercept ? s -> {} : null, additionalArgs);
	}

	// TODO: We will probably need some kind of watchdog
	// This cannot run in parallel
	public static synchronized void executeCode(String code, Consumer<String> consoleInterceptor, String... additionalArgs) {
		try {
			if (consoleInterceptor != null)
				System.setOut(new PrintStream(new CustomOutputStream(System.out, consoleInterceptor)));

			String[] args = new String[additionalArgs.length + 2];

			for (int i = 0; i < additionalArgs.length; i++)
				args[i] = additionalArgs[i];

			args[additionalArgs.length] = "-s";
			args[additionalArgs.length + 1] = code;
			DMLScript.executeScript(args);

		} catch (Exception e) {
			e.printStackTrace();
		}

		if (consoleInterceptor != null)
			System.setOut(origPrintStream);
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
			// Handle the byte 'b', or you can write to any custom destination
			//ps.print((char) b); // Example: redirect to System.err
		}

		@Override
		public void write(byte[] b, int off, int len) {
			for (int i = off; i < off + len; i++) {
				write(b[i]);
			}
		}
	}
}
