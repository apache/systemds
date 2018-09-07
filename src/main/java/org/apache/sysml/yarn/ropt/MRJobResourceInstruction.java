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

package org.apache.sysml.yarn.ropt;

import org.apache.sysml.runtime.instructions.MRJobInstruction;

/**
 * The purpose of this class is to encode the yarn mapred memory configuration into 
 * the generated runtime plan in order to take this information into account when
 * costing runtime plans. Having a subclass of MRJobInstructions allows for minimal 
 * interference with existing packages or costing in non-Yarn settings.
 * 
 */
public class MRJobResourceInstruction extends MRJobInstruction
{
	
	private long _maxMRTasks = -1;
	
	public MRJobResourceInstruction( MRJobInstruction that ) 
		throws IllegalArgumentException, IllegalAccessException
	{
		super( that );
	}
	
	public void setMaxMRTasks( long max )
	{
		_maxMRTasks = max;
	}
	
	public long getMaxMRTasks()
	{
		return _maxMRTasks;
	}

}
