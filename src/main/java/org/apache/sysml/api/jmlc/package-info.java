/**
 * Java Machine Learning Connector (JMLC) API
 * 
 * <p>
 * The Java Machine Learning Connector (JMLC) API is a programmatic interface
 * for interacting with SystemML in an embedded fashion. To use JMLC, the small
 * footprint "in-memory" SystemML jar file is included on the
 * classpath of the Java application, since JMLC invokes SystemML in an
 * existing Java Virtual Machine. Because of this, JMLC allows access to
 * SystemML's optimizations and fast linear algebra, but the bulk performance
 * gain from running SystemML on a large Spark or Hadoop cluster is not
 * available. However, this embeddable nature allows SystemML to be part of a
 * production pipeline for tasks such as scoring.
 * 
 * <p>
 * JMLC is patterned after JDBC.
 * 
 * <p>
 * For examples, please see the following:
 * <ul> 
 *   <li>JMLC JUnit test cases (org.apache.sysml.test.integration.functions.jmlc)</li>
 *   <li><a target="_blank" href="http://apache.github.io/incubator-systemml/jmlc.html">JMLC section
 *   of SystemML online documentation</li>
 * </ul>
 */
package org.apache.sysml.api.jmlc;
