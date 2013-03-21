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

import javax.management.Attribute;
import javax.management.AttributeChangeNotification;
import javax.management.DynamicMBean;
import javax.management.JMX;
import javax.management.MBeanServerConnection;
import javax.management.Notification;
import javax.management.NotificationListener;
import javax.management.ObjectName;
import javax.management.remote.JMXConnector;
import javax.management.remote.JMXConnectorFactory;
import javax.management.remote.JMXServiceURL;

import org.apache.log4j.Logger;
import org.apache.log4j.jmx.HierarchyDynamicMBean;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;


public class Client {
	private static final Log LOG = LogFactory.getLog(Client.class.getName());
	/**
     * Inner class that will handle the notifications.
     */
    public static class ClientListener implements NotificationListener {
        public void handleNotification(Notification notification,
                                       Object handback) {
            LOG.info("Received notification: ClassName: " + notification.getClass().getName());
            LOG.info("Source: " + notification.getSource());
            LOG.info("Type: " + notification.getType());
            LOG.info("Message: " + notification.getMessage());
            if (notification instanceof AttributeChangeNotification) {
                AttributeChangeNotification attrChangeNotif =
                    (AttributeChangeNotification) notification;
                LOG.info("Change Attribute: " + attrChangeNotif.getAttributeName());
                LOG.info("AttributeType: " + attrChangeNotif.getAttributeType() + " NewValue: " + attrChangeNotif.getNewValue() + " OldValue: " + attrChangeNotif.getOldValue());
            }
        }
    }
    
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

    	// Create listener
    	// TODO: This part probably is not important
    	ClientListener listener = new ClientListener();

    	// Get an MBeanServerConnection
    	MBeanServerConnection mbServerConn = jmxConnector.getMBeanServerConnection();

    	// Get domains from MBeanServer
    	// TODO: This part probably is not important
/*    	String domains[] = mbServerConn.getDomains();
    	Arrays.sort(domains);
    	for (String domain : domains) {
    		LOG.info("Domain = " + domain);
    	}*/

    	// ----------------------
    	// Manage the Hello MBean
    	// ----------------------
    	// Construct the ObjectName for the Hello MBean
    	//
    	ObjectName mbeanName = new ObjectName("log4j:hiearchy=default");

    	// Create a dedicated proxy for the MBean instead of going 
    	// directly via the MBean server connection
    	DynamicMBean mbeanProxy = JMX.newMBeanProxy(mbServerConn, mbeanName, DynamicMBean.class, true);

    	// Add notification listener on Hello MBean
    	// TODO: This probably is not important
    	mbServerConn.addNotificationListener(mbeanName, listener, null, null);

    	// Set attribute at the remote JMX server
    	//Logger rootLogger = Logger.getLogger("com.ibm.biginsights.bigsql");
    	//mbeanProxy.addLoggerMBean(rootLogger.getName());
    	Attribute attr = new Attribute("priority", new String("DEBUG"));
    	mbeanProxy.setAttribute(attr);


    	jmxConnector.close();
    }

    private static void sleep(int millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }


}
