package com.ibm.bi.dml.packagesupport;

import java.util.ArrayList;

/**
 * Sample package function that takes one input of each kind and produces the
 * very same output.
 * 
 * 
 * 
 */
public class TestPackageFunctionWorker extends PackageFunction {

 
	private static final long serialVersionUID = 8568116486233401409L;
	ArrayList<FIO> outputs;

	@Override
	public void execute() {

		System.out.println("Woo Hoo! Package function running on worker");

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
