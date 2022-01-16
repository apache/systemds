package org.apache.sysds.runtime.iogen.template;

import org.apache.sysds.runtime.iogen.CustomProperties;

public class TemplateCodeGenFrame extends TemplateCodeGenBase {

	public TemplateCodeGenFrame(CustomProperties properties, String className) {
		super(properties, className);
	}

	@Override public String generateCodeJava() {
		return null;
	}

	@Override public String generateCodeCPP() {
		return null;
	}
}
