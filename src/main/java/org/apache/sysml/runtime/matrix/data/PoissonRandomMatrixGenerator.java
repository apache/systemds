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
import org.apache.sysml.runtime.util.PoissonPRNGenerator;

public class PoissonRandomMatrixGenerator extends RandomMatrixGenerator {

	private double _mean=1.0;
	
	PoissonRandomMatrixGenerator(String pdf, int r, int c, int rpb, int cpb, double sp, double mean) throws DMLRuntimeException 
	{
		_mean = mean;
		init(pdf, r, c, rpb, cpb, sp, Double.NaN, Double.NaN);
		setupValuePRNG();
	}
	
	@Override
	protected void setupValuePRNG() throws DMLRuntimeException
	{
		if(_pdf.equalsIgnoreCase(LibMatrixDatagen.RAND_PDF_POISSON))
		{
			if(_mean <= 0)
				throw new DMLRuntimeException("Invalid parameter (" + _mean + ") for Poisson distribution.");
			
			_valuePRNG = new PoissonPRNGenerator(_mean);
		}
		else
			throw new DMLRuntimeException("Expecting a Poisson distribution (pdf = " + _pdf);
	}
	

}
