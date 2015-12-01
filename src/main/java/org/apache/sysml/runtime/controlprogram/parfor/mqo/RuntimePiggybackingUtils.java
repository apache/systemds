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

package org.apache.sysml.runtime.controlprogram.parfor.mqo;

import java.io.IOException;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;

public class RuntimePiggybackingUtils 
{

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
