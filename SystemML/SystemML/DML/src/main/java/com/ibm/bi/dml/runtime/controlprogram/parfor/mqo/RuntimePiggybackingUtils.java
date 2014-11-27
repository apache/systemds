/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.io.IOException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.yarn.ropt.YarnClusterAnalyzer;

public class RuntimePiggybackingUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	/**
	 * In general the cluster utilization reflects the percentage of
	 * currently used resources relative to maximum resources.
	 * 
	 * On MR1, we compute this by the number of occupied and maxmimum
	 * map/reduce slots. 
	 * On YARN, we use the memory consumption and virtual cores as an indicator of 
	 * the cluster utilization since the binary compatible API returns
	 * always a constant of 1 for occupied slots.
	 * 
	 * @return
	 * @throws IOException 
	 */
	public static double getCurrentClusterUtilization() 
		throws IOException
	{
		double util = 0;
		
		if( InfrastructureAnalyzer.isYarnEnabled() )
			util = YarnClusterAnalyzer.getClusterUtilization();
		else
			util = InfrastructureAnalyzer.getClusterUtilization(true);
		
		return util;
	}
}
