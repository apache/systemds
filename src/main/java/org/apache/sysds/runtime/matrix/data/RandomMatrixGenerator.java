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

package org.apache.sysds.runtime.matrix.data;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.IPRNGenerator;
import org.apache.sysds.runtime.util.NormalPRNGenerator;
import org.apache.sysds.runtime.util.PhiloxNormalCBPRNGenerator;
import org.apache.sysds.runtime.util.PhiloxUniformCBPRNGenerator;
import org.apache.sysds.runtime.util.PoissonPRNGenerator;
import org.apache.sysds.runtime.util.UniformPRNGenerator;

public class RandomMatrixGenerator {

	/**
	 * Types of Probability density functions
	 */
	public enum PDF {
		NORMAL, UNIFORM, POISSON, CB_UNIFORM, CB_NORMAL
	}

	PDF _pdf;
	int _rows, _cols, _blocksize;
	double _sparsity, _mean; 
	double _min, _max; 
	IPRNGenerator _valuePRNG;

	public RandomMatrixGenerator() {
		_pdf = PDF.UNIFORM;
		_rows = _cols = _blocksize = -1;
		_sparsity = 0.0;
		_min = _max = Double.NaN;
		_valuePRNG = null;
		_mean = 1.0;
	}
	
	public boolean isFullyDense() {
		return _sparsity == 1 & (_min != 0 | _max != 0);
	}

	/**
	 * Instantiates a Random number generator
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param blen   rows/cols per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 */
	public RandomMatrixGenerator(PDF pdf, int r, int c, int blen, double sp) {
		this(pdf, r, c, blen, sp, Double.NaN, Double.NaN);
	}

	/**
	 * Instantiates a Random number generator
	 * @param pdfStr probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param blen   rows/cols per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 */
	public RandomMatrixGenerator(String pdfStr, int r, int c, int blen, double sp, double min, double max) {
		init(PDF.valueOf(pdfStr.toUpperCase()), r, c, blen, sp, min, max);
	}

	/**
	 * Instantiates a Random number generator
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param blen   rows/cols per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 */
	public RandomMatrixGenerator(PDF pdf, int r, int c, int blen, double sp, double min, double max) {
		init(pdf, r, c, blen, sp, min, max);
	}

	/**
	 * Initializes internal data structures. Called by Constructor
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param blen   rows/cols per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 */
	public void init(PDF pdf, int r, int c, int blen, double sp, double min, double max) {
		_pdf = pdf;
		_rows = r;
		_cols = c;
		_blocksize = blen;
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
	 * @param blen   rows/cols per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @param mean   the poisson mean
	 */
	public RandomMatrixGenerator(PDF pdf, int r, int c, int blen, double sp, double min, double max, double mean) {
		init(pdf, r, c, blen, sp, min, max, mean);
	}

	/**
	 * Instantiates a Random number generator with a specific poisson mean
	 * @param pdf    probability density function
	 * @param r      number of rows
	 * @param c      number of columns
	 * @param blen   rows/cols per block
	 * @param sp     sparsity (0 = completely sparse, 1 = completely dense)
	 * @param min    minimum of range of random numbers
	 * @param max    maximum of range of random numbers
	 * @param mean   the poisson mean
	 */
	public void init(PDF pdf, int r, int c, int blen, double sp, double min, double max, double mean) {
		_pdf = pdf;
		_rows = r;
		_cols = c;
		_blocksize = blen;
		_sparsity = sp;
		_min = min;
		_max = max;
		_mean = mean;
		setupValuePRNG();
	}
	
	protected void setupValuePRNG() {
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
		case CB_UNIFORM:
			_valuePRNG = new PhiloxUniformCBPRNGenerator();
			break;
		case CB_NORMAL:
			_valuePRNG = new PhiloxNormalCBPRNGenerator();
			break;
		default:
			throw new DMLRuntimeException("Unsupported probability density function");
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("PRNG[");
		sb.append("pdf = "+_pdf.name()+", ");
		sb.append("rows = "+_rows+", ");
		sb.append("cols = "+_cols+", ");
		sb.append("blen = "+_blocksize+", ");
		sb.append("sparsity = "+_sparsity+", ");
		sb.append("min = "+_min+", ");
		sb.append("max = "+_max+"]");
		return sb.toString();
	}
}
