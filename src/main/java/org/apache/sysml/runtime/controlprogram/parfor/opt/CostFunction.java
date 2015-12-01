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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;


/**
 * Generic cost function of the form <code>y = f( X )</code>, where y is a TestMeasure 
 * (e.g., execution time, memory consumption) and X is a vector of input TestVariable values.
 * 
 * This is used for two different use cases (1) polynomial function, for one input parameter
 * (e.g., y=f(x): y = f_0 + f_1*x^1 + f_2*x^2 + ...) and (2) multidimensional functions 
 * (e.g., y=f(x1, x2, ...) with poynomials for all involved input parameters.
 *
 */
public class CostFunction 
{

	
	
	protected static final Log LOG = LogFactory.getLog(CostFunction.class.getName());
    
	public static final boolean PREVENT_NEGATIVE_ESTIMATES = true;
	
	private double[] _params    = null;
	private boolean  _multiDim  = false;
	
	public CostFunction( double[] params, boolean multiDim )
	{
		_params    = params;
		_multiDim  = multiDim; 
	}

	/**
	 * 
	 * @return
	 */
	public boolean isMultiDim()
	{
		return _multiDim;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public double estimate( double in )
	{
		double costs = 0;
		
		//compute the estimate for arbitrary orders of F
		if( _params != null )
			for( int i = 0; i<_params.length; i++ )
			{
				//test
				double v1 = in;
				double v2 = Math.pow(in, i);
				if( i>1 && Math.abs(Math.sqrt( v2 ) - v1) > 1.0 ) //issue if larger than 1ms or 1byte
				{
					LOG.error("Numerical stability issue: " + v1 + " vs " + v2 );
					continue;
				}
				//end test
				
				costs += _params[i] * Math.pow(in, i);
			}
		
		costs = correctEstimate(costs);
		
		return costs;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public double estimate( double[] in )  
	{
		double costs = 0;
		int len = in.length;
		
		if( _params != null )
		{
			costs = _params[0]; //intercept
		
			for( int i=0; i<len; i++ )
				costs += _params[i+1] * in[ i ];
			
			for( int i=0; i<len; i++ )
				costs += _params[len+i+1] * Math.pow(in[ i ],2);
			
			int ix=0;
			for( int j=0; j<len-1; j++ )
				for( int k=j+1; k<len; k++, ix++ )
					costs += _params[1+2*len+ix] * in[j]*in[k];
			
			//double tmp=1;
			//for( int i=0; i<len; i++ )
			//	tmp*=in[i];
			//costs += _params[_params.length-1]*tmp;
		}
	
		costs = correctEstimate(costs);
		
		return costs;
	}
	
	/**
	 * 
	 * @return
	 */
	public double[] getParams()
	{
		return _params;
	}
		
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder( "f(x) = " );
		
		//compute the estimate for arbitrary orders of F
		if( _params != null )
			for( int i = 0; i<_params.length; i++ )
			{
				if( i > 0 )
					sb.append( " + " );
				sb.append( _params[i] );
				sb.append( " * x^" );
				sb.append( i );
			}
		
		return sb.toString();		
	}
	
	/**
	 * 
	 * @param cost
	 * @return
	 */
	private double correctEstimate( double cost )
	{
		double ret = cost;
		
		//check for invalid estimates (due to polynomial functions)
		if( PREVENT_NEGATIVE_ESTIMATES )
		{
			ret = (ret < 0) ? 0 : ret;
		}
		
		return ret;
	}
}
