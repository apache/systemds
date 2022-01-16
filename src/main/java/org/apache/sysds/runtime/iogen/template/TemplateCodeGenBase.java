package org.apache.sysds.runtime.iogen.template;

import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.codegen.CodeGenBase;

public abstract class TemplateCodeGenBase {

	protected String code = "%code%";
	protected String prop = "%prop%";
	protected CustomProperties properties;
	protected String className;
	protected String javaTemplate;
	protected String cppSourceTemplate;
	protected String cppHeaderTemplate;

	protected CodeGenBase codeGenClass;

	public TemplateCodeGenBase(CustomProperties properties, String className) {
		this.properties = properties;
		this.className = className;
	}

	public abstract String generateCodeJava();

	public abstract String generateCodeCPP();
}
