/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.hops.codegen.cplan;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.io.IOUtilFunctions;

import java.io.FileInputStream;
import java.io.IOException;

public abstract class CodeTemplate {
	
	public String getTemplate() {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	/**
	 * @param sparseTemplate added to turn SparseRowVector intermediates on and off
	 */
	public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector,
		boolean scalarInput, boolean vectorVector, boolean sparseTemplate) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs,
		boolean scalarVector, boolean scalarInput, boolean vectorVector) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}
	
	public String getTemplate(CNodeTernary.TernaryType type, boolean sparse) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}
	
	public String getTemplate(CNodeUnary.UnaryType type, boolean sparse) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	public String getTemplate(CNodeUnary.UnaryType type, boolean sparse, boolean sparseTemplate) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}
	
	public static String getTemplate(String templateFileName) {
		try {
			// Change prefix to the code template file if running from jar. File were extracted to a temporary
			// directory in that case. By default we load the template from the source tree.
			if(CodeTemplate.class.getProtectionDomain().getCodeSource().getLocation().getPath().contains(".jar")) {
				if(templateFileName.contains(".java")) {
					templateFileName = templateFileName
						.replace("/java/org/apache/sysds/hops/codegen/cplan/java/", "/java/spoof/");
				}
				return (IOUtilFunctions.toString(new FileInputStream(ConfigurationManager.getDMLConfig()
					.getTextValue(DMLConfig.LOCAL_TMP_DIR) + templateFileName)));
			}
			else
				try(FileInputStream fis = new FileInputStream(
					System.getProperty("user.dir") + "/src/main" + templateFileName)) {
					return IOUtilFunctions.toString(fis);
				}
		}
		catch(IOException e) {
			System.out.println(e.getMessage());
			return null;
		}
	}

}
