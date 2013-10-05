/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

import java.util.ArrayList;

/**
 * Sample package function that takes one input of each kind and produces the
 * very same output.
 * 
 * 
 * 
 */
public class TestPackageFunctionMaster extends PackageFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
 
	private static final long serialVersionUID = -7532173158400546536L;
	ArrayList<FIO> outputs;

	@Override
	public void execute() {

		System.out.println("Woo Hoo! Package function running on master");

		outputs = new ArrayList<FIO>();

		// copy inputs into outputs
		for (int i = 0; i < this.getNumFunctionInputs(); i++) {
			FIO input = this.getFunctionInput(i);
			
			if(input instanceof Matrix)
				System.out.println("Input is matrix " + ((Matrix)input).getFilePath() + " " + ((Matrix)input).getNumRows() + " " + ((Matrix)input).getNumCols());
			if(input instanceof Scalar)
				System.out.println("Input is scalar " + ((Scalar)input).getValue());
			if(input instanceof bObject)
				System.out.println("Input is object " + ((bObject)input).toString());

			outputs.add(input);
		}

	}

	@Override
	public FIO getFunctionOutput(int pos) {

		if (outputs == null)
			throw new PackageRuntimeException("outputs should not be null!");

		if (pos >= outputs.size())
			throw new PackageRuntimeException("index out of bounds");

		return outputs.get(pos);
	}

	@Override
	public int getNumFunctionOutputs() {

		if (outputs == null)
			throw new PackageRuntimeException("outputs should not be null");

		return outputs.size();
	}

}
