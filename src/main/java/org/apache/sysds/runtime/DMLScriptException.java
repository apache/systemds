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

package org.apache.sysds.runtime;

/**
 * This exception should be thrown to flag DML Script errors, this exception is reserved for the stop instruction and
 * script related errors that an end-user should be able to address
 * 
 */
public class DMLScriptException extends DMLRuntimeException {
	private static final long serialVersionUID = 2L;

	/**
	 * Construct a DML script exception, this exception is reserved for the stop instruction and script related errors
	 * that an end-user should be able to address
	 *
	 * The DMLScrip exception is intended not to be able to throw other exceptions on. Therefore, there is only one
	 * constructor.
	 * 
	 * @param msg message
	 */
	public DMLScriptException(String msg) {
		super(msg);
	}
}
