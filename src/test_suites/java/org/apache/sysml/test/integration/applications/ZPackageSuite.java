/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.applications;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package/related subpackages into a single suite so that the Maven build
 *  won't run two of them at once. Since the DML and PyDML equivalent tests currently share the same directories,
 *  they should not be run in parallel. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	
  // .applications.dml package
  org.apache.sysml.test.integration.applications.dml.ApplyTransformDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.ArimaDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.CsplineCGDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.CsplineDSDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.GLMDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.GNMFDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.HITSDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.ID3DMLTest.class,
  org.apache.sysml.test.integration.applications.dml.L2SVMDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.LinearLogRegDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.LinearRegressionDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.MDABivariateStatsDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.MultiClassSVMDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.NaiveBayesDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.PageRankDMLTest.class,
  org.apache.sysml.test.integration.applications.dml.WelchTDMLTest.class,

  // .applications.pydml package
  org.apache.sysml.test.integration.applications.pydml.ApplyTransformPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.ArimaPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.CsplineCGPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.CsplineDSPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.GLMPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.GNMFPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.HITSPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.ID3PyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.L2SVMPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.LinearLogRegPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.LinearRegressionPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.MDABivariateStatsPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.MultiClassSVMPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.NaiveBayesPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.PageRankPyDMLTest.class,
  org.apache.sysml.test.integration.applications.pydml.WelchTPyDMLTest.class
  
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
