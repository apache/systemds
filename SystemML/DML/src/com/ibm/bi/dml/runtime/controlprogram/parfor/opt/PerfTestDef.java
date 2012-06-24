package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.DataFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.InternalTestVariable;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestVariable;

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
