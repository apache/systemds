/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.DataFormat;

/**
 * 
 * TODO extend to right as well (see PerfTestTool, currently only trained with regard to left)
 * TODO integrate mem and exec time for reuse?
 * 
 */
public class OptNodeStatistics 
{

	
	public static final long       DEFAULT_DIMENSION  = 100;
	public static final double     DEFAULT_SPARSITY   = 1.0;		
	public static final DataFormat DEFAULT_DATAFORMAT = DataFormat.DENSE;

	//operation characteristics
	private long _dim1 = -1; //rows left
	private long _dim2 = -1; //cols left
	private long _dim3 = -1; //rows right
	private long _dim4 = -1; //cols right
	
	private double     _sparsity  = -1; //sparsity left 
	private DataFormat _df        = null; //data format left
	
	
	/**
	 * Default constructor, sets all internal statistics to their respective default values.
	 */
	public OptNodeStatistics( )
	{
		_dim1     = DEFAULT_DIMENSION;
		_dim2     = DEFAULT_DIMENSION;
		_dim3     = DEFAULT_DIMENSION;
		_dim4     = DEFAULT_DIMENSION;
		_sparsity = DEFAULT_SPARSITY;
		_df       = DEFAULT_DATAFORMAT;
	}
	
	public OptNodeStatistics( long dim1, long dim2, long dim3, long dim4, double sparsity, DataFormat df )
	{
		_dim1     = dim1;
		_dim2     = dim2;
		_dim3     = dim3;
		_dim4     = dim4;
		_sparsity = sparsity;
		_df       = df;		
	}

	public long getDim1() 
	{
		return _dim1;
	}

	public void setDim1(long dim1) 
	{
		_dim1 = dim1;
	}

	public long getDim2() 
	{
		return _dim2;
	}

	public void setDim2(long dim2) 
	{
		_dim2 = dim2;
	}

	public long getDim3() 
	{
		return _dim3;
	}

	public void setDim3(long dim3) 
	{
		_dim3 = dim3;
	}

	public long getDim4() 
	{
		return _dim4;
	}

	public void setDim4(long dim4) 
	{
		_dim4 = dim4;
	}

	public double getSparsity() 
	{
		return _sparsity;
	}

	public void setSparsity(double sparsity) 
	{
		_sparsity = sparsity;
	}

	public DataFormat getDataFormat() 
	{
		return _df;
	}

	public void setDataFormat(DataFormat df) 
	{
		_df = df;
	}
}
