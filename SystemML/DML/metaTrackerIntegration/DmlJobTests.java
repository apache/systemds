/*******************************************************************************
 * IBM Confidential OCO Source Materials (C) Copyright IBM Corp. 2009, 2010 The source code for this program is not
 * published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright
 * Office.
 ******************************************************************************/
package com.ibm.metatracker.test.job;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Properties;

import org.apache.hadoop.fs.Path;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.ibm.jaql.json.type.BufferedJsonRecord;
import com.ibm.jaql.json.type.JsonValue;
import com.ibm.metatracker.api.CreateDB;
import com.ibm.metatracker.exception.MetaTrackerException;
import com.ibm.metatracker.job.Job;
import com.ibm.metatracker.job.JobInstance;
import com.ibm.metatracker.job.JobState;
import com.ibm.metatracker.job.JobStatus;
import com.ibm.metatracker.main.JobGraphState;
import com.ibm.metatracker.main.JobGraphStatus;
import com.ibm.metatracker.main.MetaTracker;
import com.ibm.metatracker.main.MetaTrackerConstants;
import com.ibm.metatracker.statemgr.DirMetaData;
import com.ibm.metatracker.statemgr.VersionMetaData;
import com.ibm.metatracker.util.FSUtils;
import com.ibm.metatracker.util.FileUtils;
import com.ibm.metatracker.util.JSONUtils;
import com.ibm.metatracker.util.PathUtils;
import com.ibm.metatracker.util.RegexFileFilter;
import com.ibm.metatracker.util.TestHarness;

/**
 * Tests of the built-in MetaTracker job that runs a SystemML script.
 * 
 */
public class DmlJobTests extends TestHarness {

	/**
	 * Name of the directory containing various input files for tests in this
	 * class.
	 */
	public static final File INPUTS_DIR = new File("testdata/dmlJobTests");

	public static final String JVM_FLAGS = "-Xmx1500m -Djaql.mapred.mode=local";

	public DmlJobTests() throws Exception {
		super();
	}

	/**
	 * main() method for when you only want to run a single test.
	 */
	public static void main(String[] args) throws Exception {

		DmlJobTests t = new DmlJobTests();

		long startTimeMs = System.currentTimeMillis();

		{
			t.setUp();

			t.singleTest();

			t.tearDown();
		}

		long elapsedMs = System.currentTimeMillis() - startTimeMs;
		double elapsedSec = elapsedMs / 1000.0;

		System.err.printf("Test completed in %1.1f sec.\n", elapsedSec);
	}

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
	}

	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {

		// In case the MetaTracker crashed without shutting down its Derby
		// connection, disconnect the Derby driver to allow later tests to
		// connect to other databases.
		try {
			DriverManager.getConnection("jdbc:derby:;shutdown=true");
		} catch (SQLException e) {
			// The shutdown command always raises a SQLException
			// See http://db.apache.org/derby/docs/10.2/devguide/
		}
		System.gc();
	}

	/**
	 * Runs a simple SystemML script 
	 */
	@Test
	public void singleTest() throws Exception {
		startTest();
		// Properties p = setupOutputDirs();

		final File CONFIG_FILE = new File(INPUTS_DIR, "singleDMLJob.json");

		// Create some temporary input/output directories for the job.
		// Create some fake root dirs for inputs, outputs, and working dirs.
		final Path workingDirRoot = new Path(getCurOutputDir(), "workingDirs");
		FSUtils.mkdirs(workingDirRoot);
		// localFS.mkdirs(workingDirRoot);

		// Directory under which we will create temporary output directories.
		final Path outputDirRoot = new Path(getCurOutputDir(), "outputDirs");
		FSUtils.mkdirs(outputDirRoot);
		// localFS.mkdirs(outputDirRoot);

		// Directory under which we will create permanent output directories.
		final Path permDirRoot = new Path(getCurOutputDir(), "permanentDirs");
		localFS.mkdirs(permDirRoot);

		// Create a single input dir and a single output dir.
		Path inputDir = new Path(outputDirRoot, "inputDir");
		//localFS.mkdirs(inputDir);
		Path outDir = new Path(outputDirRoot, "outDir");
		localFS.mkdirs(outDir);

		// Copy data file to input directory
		PathUtils.copyDirOrFile(new Path(INPUTS_DIR.getAbsolutePath() + "/dataSingle"), inputDir);
		
		BufferedJsonRecord configJson = (BufferedJsonRecord) JSONUtils
				.fileToJSON(CONFIG_FILE);

		// No additional variables to pass to this job, for now.
		HashMap<String, JsonValue> addlVars = new HashMap<String, JsonValue>();

		// Instantiate the job.

		if (log.isInfoEnabled())
			log.info(String.format("Job config record is:\n%s\n", configJson));

		Job job = new Job(configJson);

		JobInstance instance = job
				.instantiate(DirMetaData.makeDummy(workingDirRoot), makeDirSet(
						"input", inputDir), makeDirSet("output", outDir),
						emptyVersionSet(), addlVars, null);

		instance.setDefaultLogForTests();
		// Run the job to completion.
		runInstance(instance);

		// Make sure the job produced the expected output file.
		Path outFile = new Path(outputDirRoot, "outDir/variables.json");
		assertTrue(localFS.exists(outFile));
	}

	/**
	 * Set up output directories for running a metatracker instance.
	 * 
	 * @return Properties object pointing to the directory locations
	 */
	private Properties setupOutputDirs() {
		Properties p = new Properties();
		p.setProperty(MetaTrackerConstants.OUTPUT_DIR_PROP, getCurOutputDir()
				+ "/outputDirs");
		p.setProperty(MetaTrackerConstants.PERM_DIR_PROP, getCurOutputDir()
				+ "/permanentDirs");
		p.setProperty(MetaTrackerConstants.WORKING_DIR_PROP, getCurOutputDir()
				+ "/workingDirs");
		p.setProperty(MetaTrackerConstants.DB_ROOT_PROP, getCurOutputDir()
				+ "/dbRoot");
		p.setProperty(MetaTrackerConstants.COLLECTED_LOGS_DIR_PROP,
				getCurOutputDir() + "/logs");
		return p;
	}

	/**
	 * Create a temporary MetaTracker state store, then create a MetaTracker
	 * instance that points to that store.
	 * 
	 * @param p
	 *            properties object containing MetaTracker configuration
	 */
	private MetaTracker makeMetaTracker(Properties p) throws Exception {
		CreateDB.run(p);

		File dbRoot = new File(p.getProperty(MetaTrackerConstants.DB_ROOT_PROP));
		return new MetaTracker(dbRoot);
	}

	public static final long POLL_INTERVAL_MSEC = 5000;

	/**
	 * Run a single job graph until it completes or fails
	 * 
	 * @param mt
	 *            initialized MetaTracker instance
	 * @param jobGraphJSON
	 *            JSON specification of the job graph
	 * @throws MetaTrackerException
	 * @throws InterruptedException
	 */
	public void runJobGraph(MetaTracker mt, JsonValue jobGraphJSON)
			throws MetaTrackerException, InterruptedException, IOException {
		String graphName = mt.addGraph(jobGraphJSON);

		// Poll the status of the JobGraph until it finishes.
		while (true) {
			JobGraphStatus status = mt.getGraphStatus(graphName);
			JobGraphState state = status.getState();

			System.err.printf("DRIVER SCRIPT: Graph state is %s\n", state);

			if (JobGraphState.SUCCEEDED.equals(state)) {
				// Shut down the MetaTracker in a controlled manner.
				mt.shutdown();
				break;
			} else if (JobGraphState.FAILED.equals(state)) {
				mt.shutdown();
				throw new MetaTrackerException("Job graph failed");
			}

			Thread.sleep(POLL_INTERVAL_MSEC);
		}
	}

	/**
	 * Create a basic input dirs set for a single input or output directory.
	 * 
	 * @throws MetaTrackerException
	 */
	protected static HashMap<String, DirMetaData> makeDirSet(String dirName,
			Path path) throws MetaTrackerException {
		HashMap<String, DirMetaData> ret = new HashMap<String, DirMetaData>();
		ret.put(dirName, DirMetaData.makeDummy(path));
		return ret;
	}

	// /**
	// * Create a basic input dirs set for multiple directories.
	// *
	// * @throws MetaTrackerException
	// */
	// protected static HashMap<String, DirMetaData> makeDirSet(
	// Object[][] dirPathPairs, FileSystem fs) throws MetaTrackerException {
	// HashMap<String, DirMetaData> ret = new HashMap<String, DirMetaData>();
	// for (Object[] pair : dirPathPairs) {
	// String dirName = (String) pair[0];
	// Path path = (Path) pair[1];
	// ret.put(dirName, DirMetaData.makeDummy(fs, path));
	// }
	// return ret;
	// }

	// /**
	// * Create a basic input dirs set for multiple directories.
	// *
	// * @throws MetaTrackerException
	// */
	// protected static HashMap<String, DirMetaData> makeDirSet(
	// Object[][] dirPathPairs) throws MetaTrackerException {
	// HashMap<String, DirMetaData> ret = new HashMap<String, DirMetaData>();
	// for (Object[] pair : dirPathPairs) {
	// String dirName = (String) pair[0];
	// File path = (File) pair[1];
	// ret.put(dirName, DirMetaData.makeDummy(path));
	// }
	// return ret;
	// }

	public static HashMap<String, DirMetaData> emptyDirSet() {
		return new HashMap<String, DirMetaData>();
	}

	public static HashMap<String, VersionMetaData> emptyVersionSet() {
		return new HashMap<String, VersionMetaData>();
	}

	/**
	 * Run a job instance to completion.
	 * 
	 * @throws MetaTrackerException
	 */
	public static JobState runInstance(JobInstance instance)
			throws MetaTrackerException {

		// Now we can run the job.
		instance.start();

		JobStatus status = instance.getStatus();

		// Wait until the job is complete.
		while (JobState.RUNNING.equals(status.getState())) {

			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				// Keep going if we're interrupted.
			}
			status = instance.getStatus();

			System.err.printf(
					"Waiting for job to complete.  Current state is %s.\n",
					status.getState());
		}

		System.err.printf("Final job instance state is %s.\n", status
				.getState());
		return status.getState();

	}
}
