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

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.DataFormat;
import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.InternalTestVariable;
import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.TestVariable;

/**
 * Internal representation of a test configuration consisting of a logical TestMeasure,
 * a logical TestVariable as well as a DataFormat. Note that one cost function refers
 * to a statistical model of a profiling run for a combination of such a test configuration 
 * and an instruction.
 *
 */
public class PerfTestDef 
{

	
	//logical properties
	private TestMeasure    _measure;
	private TestVariable   _lvariable;
	private DataFormat     _dataformat;
	
	//physical properties
	private InternalTestVariable[] _pvariables;
	private double         _min;
	private double         _max;
	private double         _samples;
	
	public PerfTestDef( TestMeasure m, TestVariable lv, DataFormat df, InternalTestVariable pv, double min, double max, double samples )
	{
		this( m, lv, df, new InternalTestVariable[]{pv}, min, max, samples);
	}
	
	public PerfTestDef( TestMeasure m, TestVariable lv, DataFormat df, InternalTestVariable[] pv, double min, double max, double samples )
	{
		_measure  = m;
		_lvariable = lv;
		_dataformat = df;
		
		_pvariables = pv;
		_min = min;
		_max = max;
		_samples = samples;
	}
	
	public TestMeasure getMeasure()
	{
		return _measure;
	}
	
	public TestVariable getVariable()
	{
		return _lvariable;
	}
	
	public DataFormat getDataformat()
	{
		return _dataformat;
	}
	
	public InternalTestVariable[] getInternalVariables()
	{
		return _pvariables;
	}
	
	public double getMin()
	{
		return _min;
	}
	
	public double getMax()
	{
		return _max;
	}
	
	public double getNumSamples()
	{
		return _samples;
	}
}
