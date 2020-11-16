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

package org.apache.sysds.hops.codegen.cplan.cuda;

import java.io.FileInputStream;
import java.io.IOException;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;
import org.apache.sysds.runtime.codegen.SpoofCellwise;
import org.apache.sysds.runtime.io.IOUtilFunctions;


// ToDo: clean code template and load from file
public class CellWise implements CodeTemplate {

	private static final String TEMPLATE_PATH = "/cuda/spoof/cellwise.cu";

	@Override
	public String getTemplate() {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	@Override
	public String getTemplate(SpoofCellwise.CellType ct) {
		try {
			// Change prefix to the code template file if running from jar. File were extracted to a temporary
			// directory in that case. By default we load the template from the source tree.
			if(CellWise.class.getProtectionDomain().getCodeSource().getLocation().getPath().contains(".jar"))
				return(IOUtilFunctions.toString(new FileInputStream(ConfigurationManager.getDMLConfig()
						.getTextValue(DMLConfig.LOCAL_TMP_DIR) + TEMPLATE_PATH)));
			else
				return IOUtilFunctions.toString(new FileInputStream(System.getProperty("user.dir") +
						"/src/main" + TEMPLATE_PATH));
		}
		catch(IOException e) {
			System.out.println(e.getMessage());
			return null;
		}
	}

	@Override
	public String getTemplate(CNodeUnary.UnaryType type, boolean sparse) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	@Override
	public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector, boolean scalarInput) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	@Override
	public String getTemplate(CNodeTernary.TernaryType type, boolean sparse) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}
}
