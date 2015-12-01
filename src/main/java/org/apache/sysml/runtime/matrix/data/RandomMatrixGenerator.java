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

package org.apache.sysml.runtime.matrix.data;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.NormalPRNGenerator;
import org.apache.sysml.runtime.util.PRNGenerator;
import org.apache.sysml.runtime.util.PoissonPRNGenerator;
import org.apache.sysml.runtime.util.UniformPRNGenerator;

public class RandomMatrixGenerator {
	
	String _pdf;
	int _rows, _cols, _rowsPerBlock, _colsPerBlock;
	double _sparsity, _mean; 
	double _min, _max; 
	PRNGenerator _valuePRNG;
	//Well1024a _bigrand; Long _bSeed;

	public RandomMatrixGenerator() 
	{
		_pdf = "";
		_rows = _cols = _rowsPerBlock = _colsPerBlock = -1;
		_sparsity = 0.0;
		_min = _max = Double.NaN;
		_valuePRNG = null;
		_mean = 1.0;
	}
	
	public RandomMatrixGenerator(String pdf, int r, int c, int rpb, int cpb, double sp) throws DMLRuntimeException 
	{
		this(pdf, r, c, rpb, cpb, sp, Double.NaN, Double.NaN);
	}
	
	public RandomMatrixGenerator(String pdf, int r, int c, int rpb, int cpb, double sp, double min, double max) throws DMLRuntimeException 
	{
		init(pdf, r, c, rpb, cpb, sp, min, max);
	}
	
	public void init(String pdf, int r, int c, int rpb, int cpb, double sp, double min, double max) throws DMLRuntimeException 
	{
		_pdf = pdf;
		_rows = r;
		_cols = c;
		_rowsPerBlock = rpb;
		_colsPerBlock = cpb;
		_sparsity = sp;
		_min = min;
		_max = max;
		
		setupValuePRNG();
	}
	
	public RandomMatrixGenerator(String pdf, int r, int c, int rpb, int cpb, double sp, double min, double max, double mean) throws DMLRuntimeException 
	{
		init(pdf, r, c, rpb, cpb, sp, min, max, mean);
	}
	
	public void init(String pdf, int r, int c, int rpb, int cpb, double sp, double min, double max, double mean) throws DMLRuntimeException 
	{
		_pdf = pdf;
		_rows = r;
		_cols = c;
		_rowsPerBlock = rpb;
		_colsPerBlock = cpb;
		_sparsity = sp;
		_min = min;
		_max = max;
		_mean = mean;
		setupValuePRNG();
	}
	
	protected void setupValuePRNG() throws DMLRuntimeException 
	{
		if ( _pdf.equalsIgnoreCase(LibMatrixDatagen.RAND_PDF_NORMAL) ) 
			_valuePRNG = new NormalPRNGenerator();
		else if ( _pdf.equalsIgnoreCase(LibMatrixDatagen.RAND_PDF_UNIFORM) ) 
			_valuePRNG = new UniformPRNGenerator();
		else if ( _pdf.equalsIgnoreCase(LibMatrixDatagen.RAND_PDF_POISSON) ) 
		{
			if(_mean <= 0)
				throw new DMLRuntimeException("Invalid parameter (" + _mean + ") for Poisson distribution.");
			_valuePRNG = new PoissonPRNGenerator(_mean);
		}
	}
}
