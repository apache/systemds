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

package com.ibm.bi.dml.test.integration;

import java.util.HashMap;

import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * <p>
 *  A configuration class which can be used to organize different tests in a test class. A test configuration can hold
 *  information about all the output files which can be used for comparisons and have to be removed after the test run.
 *  It can also handle variables for a DML script.
 * </p>
 * 
 * 
 */
public class TestConfiguration 
{

	
    
    /** directory where the test can be found */
    private String testDirectory = null;
    
    /** name of the test script */
    private String testScript = null;
    
    /** list of output files which are produced by the test */
    private String[] outputFiles = null;
    
    /** list of variables which can be replaced in the script */
    private HashMap<String, String> variables = new HashMap<String, String>();
    
    
    /**
     * <p>
     *  Creates a new test configuration with the name of the test script and the output files which are produced by
     *  the test.
     * </p>
     * 
     * @param testScript test script
     * @param outputFiles output files
     * @deprecated use TestConfiguration(String, String, String[]) instead
     */
    @Deprecated
    public TestConfiguration(String testScript, String[] outputFiles) {
        this.testScript = testScript;
        this.outputFiles = outputFiles;
    }
    
    /**
     * <p>
     *  Creates a new test configuration with the directory where the test data can be found, the name of the test
     *  script and the output files which are produced by the test.
     * </p>
     * 
     * @param testDirectory test directory
     * @param testScript test script
     * @param outputFiles output files
     */
    public TestConfiguration(String testDirectory, String testScript, String[] outputFiles) {
        this.testDirectory = testDirectory;
        this.testScript = testScript;
        this.outputFiles = outputFiles;
    }
    
    /**
     * <p>Adds a variable to the test configuration.</p>
     * 
     * @param variableName variable name
     * @param variableValue variable value
     */
    public void addVariable(String variableName, String variableValue) {
        variables.put(variableName, variableValue);
    }
    
    /**
     * <p>Adds a variable to the test configuration.</p>
     * 
     * @param variableName variable name
     * @param variableValue variable value
     */
    public void addVariable(String variableName, Boolean variableValue) {
        variables.put(variableName, variableValue.toString());
    }

    /**
     * <p>Adds a variable to the test configuration.</p>
     * 
     * @param variableName variable name
     * @param variableValue variable value
     */
    public void addVariable(String variableName, double variableValue) {
    	variables.put(variableName, TestUtils.getStringRepresentationForDouble(variableValue));
    }
    
    /**
     * <p>Adds a variable to the test configuration.</p>
     * 
     * @param variableName variable name
     * @param variableValue variable value
     */
    public void addVariable(String variableName, long variableValue) {
        variables.put(variableName, Long.toString(variableValue));
    }
    
    /**
     * <p>Provides the directory which contains the test data.</p>
     * 
     * @return test directory
     */
    public String getTestDirectory() {
        return testDirectory;
    }
    
    /**
     * <p>Provides the name of the test script.</p>
     * 
     * @return test script name
     */
    public String getTestScript() {
        return testScript;
    }
    
    /**
     * <p>Provides the list of specified output files for the test.</p>
     * 
     * @return output files
     */
    public String[] getOutputFiles() {
        return outputFiles;
    }
    
    /**
     * <p>Provides the list of the specified variables with their replacements.</p>
     * 
     * @return variables
     */
    public HashMap<String, String> getVariables() {
        return variables;
    }
    
}
