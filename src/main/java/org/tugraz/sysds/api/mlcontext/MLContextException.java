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

package org.tugraz.sysds.api.mlcontext;

/**
 * Uncaught exception representing SystemDS exceptions that occur through the
 * MLContext API.
 *
 */
public class MLContextException extends RuntimeException {

	private static final long serialVersionUID = 1842275827863526536L;
	private boolean suppressStacktrace = false;

	public MLContextException() {
		super();
	}

	public MLContextException(String message, Throwable cause) {
		super(message, cause);
	}

	public MLContextException(String message) {
		super(message);
	}

	public MLContextException(Throwable cause) {
		super(cause);
	}

	/**
	 * Generate an exception and optionally suppress the stacktrace. This can be
	 * useful in an environment such as a Spark Shell in certain situations
	 * where a stacktrace may be extraneous.
	 *
	 * @param message
	 *            the exception message
	 * @param suppressStacktrace
	 *            {@code true} to suppress stacktrace, {@code false} otherwise
	 */
	public MLContextException(String message, boolean suppressStacktrace) {
		super(message, null, suppressStacktrace, !suppressStacktrace);
		this.suppressStacktrace = suppressStacktrace;
	}

	@Override
	public String toString() {
		if (suppressStacktrace) {
			return getLocalizedMessage();
		} else {
			return super.toString();
		}
	}

}
