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

package org.apache.sysml.runtime.matrix.data;

/**
 * Helper class for external key/value exchange.
 * 
 */
public class IJV
{
	
	public int i=-1;
	public int j=-1;
	public double v=0;
	
	public IJV()
	{
		
	}
	
	public IJV(int i, int j, double v)
	{
		set(i, j, v);
	}
	
	public void set(int i, int j, double v)
	{
		this.i = i;
		this.j = j;
		this.v = v;
	}
	
	public String toString()
	{
		return "("+i+", "+j+"): "+v;
	}
}
