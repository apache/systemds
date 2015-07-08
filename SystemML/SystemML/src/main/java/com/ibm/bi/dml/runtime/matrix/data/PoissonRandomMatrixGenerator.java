/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.util.PoissonPRNGenerator;

public class PoissonRandomMatrixGenerator extends RandomMatrixGenerator {

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
            "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
