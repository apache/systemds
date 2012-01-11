package dml.packagesupport;

import org.nimble.task.AbstractWrapperTaskMasterSystemSerializable;

/**
 * Control node package functions will be wrapped using this task so that they
 * can be executed directly inside NIMBLE.
 * 
 * @author aghoting
 * 
 */

public class WrapperTaskForControlNode extends
		AbstractWrapperTaskMasterSystemSerializable {

	private static final long serialVersionUID = -6662700672478635976L;
	PackageFunction functionObject;

	/**
	 * Constructor that takes a package function object as parameter.
	 * 
	 * @param functionObject
	 */
	public WrapperTaskForControlNode(PackageFunction functionObject) {
		this.functionObject = functionObject;
	}

	@Override
	public boolean execute() {

		if (functionObject == null)
			throw new PackageRuntimeException(
					"Function object should not be null");

		functionObject.setDAGQueue(this.getDAGQueue());

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
		return "WrapperTaskControlNode";
	}

}
