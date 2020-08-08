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

package org.apache.sysds.runtime.controlprogram.federated;

/**
 * Exception to throw when an exception occurs in FederatedWorkerHandler during handling of FederatedRequest. The
 * purpose of FederatedWorkerHandlerException is to propagate useful information from the federated workers to the
 * federated master without exposing details that are usually included in exceptions, for instance name of files that
 * were not found or data points that could not be handled correctly.
 */
public class FederatedWorkerHandlerException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * Create new instance of FederatedWorkerHandlerException with a message.
	 * 
	 * @param msg message describing the exception
	 */
	public FederatedWorkerHandlerException(String msg) {
		super(msg);
	}
	
	public FederatedWorkerHandlerException(String msg, Throwable t) {
		super(msg, t);
	}
}
