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

package org.apache.sysml.runtime.functionobjects;

import java.io.Serializable;

/**
 * Abstraction for comparison (relational) operators in order to 
 * force a proper implementation by all relevant subclasses.
 */
public abstract class ValueComparisonFunction extends ValueFunction implements Serializable
{
	private static final long serialVersionUID = 6021132561216734747L;
	
	public abstract boolean compare(double in1, double in2);
	
	public abstract boolean compare(long in1, long in2);
	
	public abstract boolean compare(boolean in1, boolean in2);
	
	public abstract boolean compare(String in1, String in2);
}
