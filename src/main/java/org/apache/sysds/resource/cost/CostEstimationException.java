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

package org.apache.sysds.resource.cost;

/**
 * Exception thrown when the cost estimation gets in
 * a state that should not raise runtime a exception.
 * Such exception is to be raised only in the following case:
 * Local memory is not sufficient for the estimated caching
 */
public class CostEstimationException extends Exception {
	private static final long serialVersionUID = -6709101762468084495L;

	public CostEstimationException(String message) {
		super(message);
	}
}
