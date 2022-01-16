package org.apache.sysds.runtime.iogen.codegen;

import org.apache.sysds.runtime.iogen.CustomProperties;

public abstract class CodeGenBase {

	protected CustomProperties properties;
	protected String className;

	public CodeGenBase(CustomProperties properties, String className) {
		this.properties = properties;
		this.className = className;
	}

	public abstract String generateCodeJava();

	public abstract String generateCodeCPP();
}
