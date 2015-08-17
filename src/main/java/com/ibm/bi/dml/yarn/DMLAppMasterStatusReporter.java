/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.client.api.AMRMClient;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;

public class DMLAppMasterStatusReporter extends Thread
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
