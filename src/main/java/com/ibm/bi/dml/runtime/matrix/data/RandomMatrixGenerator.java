/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.util.NormalPRNGenerator;
import com.ibm.bi.dml.runtime.util.PRNGenerator;
import com.ibm.bi.dml.runtime.util.PoissonPRNGenerator;
import com.ibm.bi.dml.runtime.util.UniformPRNGenerator;

public class RandomMatrixGenerator {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
            "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
