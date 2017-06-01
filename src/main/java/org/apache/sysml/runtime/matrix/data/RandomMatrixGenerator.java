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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.NormalPRNGenerator;
import org.apache.sysml.runtime.util.PRNGenerator;
import org.apache.sysml.runtime.util.PoissonPRNGenerator;
import org.apache.sysml.runtime.util.UniformPRNGenerator;

public class RandomMatrixGenerator {

	/**
	 * Types of Probability density functions
	 */
	enum PDF {
		NORMAL, UNIFORM, POISSON
	}

	PDF _pdf;
	int _rows, _cols, _rowsPerBlock, _colsPerBlock;
	double _sparsity, _mean; 
	double _min, _max; 
	PRNGenerator _valuePRNG;
	//Well1024a _bigrand; Long _bSeed;

	public RandomMatrixGenerator() 
	{
		_pdf = PDF.UNIFORM;
		_rows = _cols = _rowsPerBlock = _colsPerBlock = -1;
		_sparsity = 0.0;
		_min = _max = Double.NaN;
		_valuePRNG = null;
		_mean = 1.0;
	}

	/**
	 * Instantiates a Random number generator
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param rpb    rows per block
	 * @param cpb    columns per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @throws DMLRuntimeException if error
	 */
	public RandomMatrixGenerator(PDF pdf, int r, int c, int rpb, int cpb, double sp) throws DMLRuntimeException
	{
		this(pdf, r, c, rpb, cpb, sp, Double.NaN, Double.NaN);
	}

	/**
	 * Instantiates a Random number generator
	 * @param pdfStr probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param rpb    rows per block
	 * @param cpb    columns per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @throws DMLRuntimeException if error
	 */
	public RandomMatrixGenerator(String pdfStr, int r, int c, int rpb, int cpb, double sp, double min, double max) throws DMLRuntimeException
	{
		init(PDF.valueOf(pdfStr.toUpperCase()), r, c, rpb, cpb, sp, min, max);
	}

	/**
	 * Instantiates a Random number generator
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param rpb    rows per block
	 * @param cpb    columns per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @throws DMLRuntimeException if error
	 */
	public RandomMatrixGenerator(PDF pdf, int r, int c, int rpb, int cpb, double sp, double min, double max) throws DMLRuntimeException
	{
		init(pdf, r, c, rpb, cpb, sp, min, max);
	}

	/**
	 * Initializes internal data structures. Called by Constructor
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param rpb    rows per block
	 * @param cpb    columns per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @throws DMLRuntimeException if error
	 */
	public void init(PDF pdf, int r, int c, int rpb, int cpb, double sp, double min, double max) throws DMLRuntimeException
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

	/**
	 * Instantiates a Random number generator with a specific poisson mean
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param rpb    rows per block
	 * @param cpb    columns per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @param mean   the poisson mean
	 * @throws DMLRuntimeException if error
	 */
	public RandomMatrixGenerator(PDF pdf, int r, int c, int rpb, int cpb, double sp, double min, double max, double mean) throws DMLRuntimeException
	{
		init(pdf, r, c, rpb, cpb, sp, min, max, mean);
	}

	/**
	 * Instantiates a Random number generator with a specific poisson mean
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param rpb    rows per block
	 * @param cpb    columns per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @param mean   the poisson mean
	 * @throws DMLRuntimeException if error
	 */
	public void init(PDF pdf, int r, int c, int rpb, int cpb, double sp, double min, double max, double mean) throws DMLRuntimeException
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
		switch (_pdf) {
		case NORMAL:
			_valuePRNG = new NormalPRNGenerator();
			break;
		case UNIFORM:
			_valuePRNG = new UniformPRNGenerator();
			break;
		case POISSON:
			if(_mean <= 0)
				throw new DMLRuntimeException("Invalid parameter (" + _mean + ") for Poisson distribution.");
			_valuePRNG = new PoissonPRNGenerator(_mean);
			break;
		default:
			throw new DMLRuntimeException("Unsupported probability density function");
		}
	}
}
