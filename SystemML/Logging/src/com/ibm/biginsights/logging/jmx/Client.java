/*
 * Copyright (C) IBM Corp. 2013.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package com.ibm.biginsights.logging.jmx;

import javax.management.JMX;
import javax.management.MBeanServerConnection;
import javax.management.ObjectName;
import javax.management.remote.JMXConnector;
import javax.management.remote.JMXConnectorFactory;
import javax.management.remote.JMXServiceURL;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;


public class Client {
	private static final Log LOG = LogFactory.getLog(Client.class.getName());
    
	/**
	 * @param args
	 * TODO: Add hostname, port, logger name, log level as input parameters
	 */
    public static void main(String[] args) throws Exception {
    	
    	// Create an RMI connector client and
    	// connect it to the RMI connector server
    	// TODO: add host/port support in the future
    	JMXServiceURL url =
    			new JMXServiceURL("service:jmx:rmi:///jndi/rmi://localhost:10011/jmxrmi");
    	JMXConnector jmxConnector = JMXConnectorFactory.connect(url, null);

    	// Get an MBeanServerConnection
    	MBeanServerConnection mbServerConn = jmxConnector.getMBeanServerConnection();


    	// ----------------------
    	// Manage the Log4jConfig MXBean
    	// ----------------------

    	ObjectName mbeanName = new ObjectName("com.ibm.biginsights.logging.jmx:type=Log4jConfig");

    	// Create a dedicated proxy for the MBean instead of going 
    	// directly via the MBean server connection
    	Log4jConfigMXBean mbeanProxy = JMX.newMBeanProxy(mbServerConn, mbeanName, Log4jConfigMXBean.class, true);

    	// Set attribute at the remote JMX server
    	mbeanProxy.setLogLevel("com.ibm.biginsights.bigsql", "INFO");


    	jmxConnector.close();
    }



}
