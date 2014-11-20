/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf;

import org.nimble.task.AbstractWrapperTaskWorkerSystemSerializable;

/**
 * Worker node package functions will be wrapped using this task so that they
 * can be executed directly inside NIMBLE.
 * 
 * 
 * 
 */

public class WrapperTaskForWorkerNode extends
		AbstractWrapperTaskWorkerSystemSerializable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -5425643290141531926L;
	PackageFunction functionObject;

	/**
	 * Constructor that takes a package function object as parameter.
	 * 
	 * @param functionObject
	 */
	public WrapperTaskForWorkerNode(PackageFunction functionObject) {
		this.functionObject = functionObject;
	}

	@Override
	public boolean execute() {

		if (functionObject == null)
			throw new PackageRuntimeException(
					"Function object should not be null");

		functionObject.execute();

		return true;

	}

	/**
	 * Method to return the updated package function object after it is
	 * processed.
	 * 
	 * @return
	 */
	public PackageFunction getUpdatedPackageFunction() {
		return functionObject;
	}

	
	@Override
	public String getTaskName()
	{
		return "WrapperTaskWorkerNode";
	}
}
