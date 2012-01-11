package dml.packagesupport;

import org.nimble.task.AbstractWrapperTaskWorkerSystemSerializable;

/**
 * Worker node package functions will be wrapped using this task so that they
 * can be executed directly inside NIMBLE.
 * 
 * @author aghoting
 * 
 */

public class WrapperTaskForWorkerNode extends
		AbstractWrapperTaskWorkerSystemSerializable {

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
