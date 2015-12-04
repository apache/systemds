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

package org.apache.sysml.yarn;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.client.api.AMRMClient;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;

public class DMLAppMasterStatusReporter extends Thread
{
	
	public static long DEFAULT_REPORT_INTERVAL = 5000;
	private static final Log LOG = LogFactory.getLog(DMLAppMasterStatusReporter.class);
	
	
	private AMRMClient<ContainerRequest> _rmClient;
	private long _interval; //in ms
	private boolean _stop;	
	
	
	public DMLAppMasterStatusReporter(AMRMClient<ContainerRequest> rmClient, long interval) 
	{
		_rmClient = rmClient;
		_interval = interval;
		_stop = false;
	}
	
	public void stopStatusReporter()
	{
		_stop = true;
	}
	
	@Override
	public void run() 
	{
		while( !_stop ) 
		{
			try
			{
				//report status (serves as heatbeat to RM)
				_rmClient.allocate(0);

				//sleep for interval ms until next report
				Thread.sleep( _interval );
			}
			catch(Exception ex)
			{
				LOG.error("Failed to report status to ResourceManager.", ex);
			}
		}
	}
}
