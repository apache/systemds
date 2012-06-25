/*******************************************************************************
 * 
 * IBM Confidential
 * 
 * OCO Source Materials
 * 
 * (C) Copyright IBM Corp. 2009, 2010
 * 
 * The source code for this program is not published or
 * 
 * otherwise divested of its trade secrets, irrespective of
 * 
 * what has been deposited with the U.S. Copyright Office.
 * 
 ******************************************************************************/
package com.ibm.metatracker.jobdef;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.jaql.json.type.JsonRecord;
import com.ibm.jaql.json.type.JsonString;
import com.ibm.jaql.json.type.JsonValue;
import com.ibm.metatracker.exception.MetaTrackerException;
import com.ibm.metatracker.job.JavaJobInstance;
import com.ibm.metatracker.job.JobState;
import com.ibm.metatracker.job.JobStatus;
import com.ibm.metatracker.util.FileSystemFactory;
import com.ibm.metatracker.util.JSONUtils;
import com.ibm.metatracker.util.LoggerUtils;

import dml.DMLScriptCmd;

/**
 * Built-in job type for running a SystemML DML scripts.
 * 
 */

public class DmlJob extends JavaJobInstance {

	private static final String INPUT_DIR_NAME = "input";
	private static final String OUTPUT_DIR_NAME = "output";
	private static final String DML_PATTERN_VAR = "dml.pattern";
	private static final String DML_SCRIPTS_DIR_VAR = "dml.scripts.dir";
	private static final String DML_SCRIPT_NAME_VAR = "dml.script.name";
	private static final String DML_ARGS_START_VAR = "$";
	private static final String DML_SWITCH_VAR = "dml.switch";
	private static final JsonString DML_SWITCH_FIELD_VAR = new JsonString(
			"switch.field");
	private static final JsonString DML_SWITCH_CASES_VAR = new JsonString(
			"cases");
	private static final String DML_ARGS_POUND_FILE_VAR = new String("#FILE#");
	private static final String DML_ARGS_POUND_FILE1_VAR = new String("#FILE1#");
	private static final String DML_ARGS_POUND_FILE2_VAR = new String("#FILE2#");
	private static final JsonString DML_ARGS_MODE_IN = new JsonString("in");
	private static final JsonString DML_ARGS_MODE_OUT = new JsonString("out");
	private static final JsonString DML_ARGS_MODE_VAR = new JsonString("mode");
	private static final JsonString DML_ARGS_VALUE_VAR = new JsonString("value");
	private static final JsonString MTD_DESCR = new JsonString("description");

	static public enum Pattern {
		single, forEachFile, univariate, bivariate
	};

	static public HashMap<String, Pattern> patternList;
	static {
		patternList = new HashMap<String, Pattern>();
		patternList.put("single", Pattern.single);
		patternList.put("forEachFile", Pattern.forEachFile);
		patternList.put("univariate", Pattern.univariate);
		patternList.put("bivariate", Pattern.bivariate);
	};

	protected final Log processLog = LoggerUtils.getMTLogger(DmlJob.class);

	/*
	 * METHODS
	 */

	String[] createArgsList(Path inDirPath, Path outDirPath,
			Map<String, JsonValue> margs, String poundName1, String poundName2)
			throws MetaTrackerException {
		Vector<String> args = new Vector<String>();
		int idx;

		// populate args[0] with -f script name

		args.add(0, "-f "
				+ new Path(margs.get(DML_SCRIPTS_DIR_VAR).toString(), margs
						.get(DML_SCRIPT_NAME_VAR).toString()).toString());

		// populate rest of args list with $ entries

		String arg, value;

		for (Entry<String, JsonValue> v : margs.entrySet()) {

			// only use JSON fields that start w/ $
			if (v.getKey().startsWith(DML_ARGS_START_VAR)) {

				JsonRecord jrec = (JsonRecord) v.getValue();
				value = jrec.get(DML_ARGS_VALUE_VAR) == null ? null : jrec.get(
						DML_ARGS_VALUE_VAR).toString();

				// replace #FILE1, #FILE2#, and #FILE# in value with poundName
				// if present; ORDER IS IMPORTANT!!
				if (poundName2 != null) {
					value = value.replace(DML_ARGS_POUND_FILE2_VAR, poundName2);
				}
				;
				if (poundName1 != null) {
					value = value.replace(DML_ARGS_POUND_FILE1_VAR, poundName1);
					value = value.replace(DML_ARGS_POUND_FILE_VAR, poundName1);
				}
				;

				JsonString mode = (JsonString) jrec.get(DML_ARGS_MODE_VAR);

				if (DML_ARGS_MODE_IN.equals(mode)) {
					arg = new Path(inDirPath, value).toString();
				} else if (DML_ARGS_MODE_OUT.equals(mode)) {
					arg = new Path(outDirPath, value).toString();
				} else {
					arg = value;
				}

				// Grow args vector if necessary

				idx = Integer.parseInt(v.getKey().substring(1));
				if (idx >= args.size()) {
					args.setSize(idx + 1);
					// args.ensureCapacity(idx + 1);

				}

				args.setElementAt(arg, idx);
			}
		}

		String[] sargs = new String[args.size()];
		args.toArray(sargs);
		return sargs;
	}

	// Convert Map w/ JsonString key to Map w/ String key

	Map<String, JsonValue> JMapToSMap(Map<JsonString, JsonValue> jmap) {
		TreeMap<String, JsonValue> smap = new TreeMap<String, JsonValue>();
		for (Entry<JsonString, JsonValue> e : jmap.entrySet()) {
			smap.put(e.getKey().toString(), e.getValue());
		}
		return smap;
	}

	@Override
	protected void run() throws MetaTrackerException, Exception {

		logStatusUpdate(makeStatus("Begin MT DmlJob Run."));

		// Input/Output Dir URIs
		Path inDirPath = getInputDir(INPUT_DIR_NAME);
		Path outDirPath = getOutputDir(OUTPUT_DIR_NAME);

		String[] args;

		switch (patternList.get(getStrVariable(DML_PATTERN_VAR))) {

		case single: {
			// create args list and invoke SystemML w/ args

			args = createArgsList(inDirPath, outDirPath, getVars(), null, null);

			// invoke SystemML w/ arguments
			logStatusUpdate(makeStatus("Run: DMLScriptCmd.main "
					+ args.toString()));

			DMLScriptCmd.main(args);

			break;
		}

		case forEachFile: {

			// Iterate over all the files in input directory: #FILE#. Skip files
			// that start with a ".", are zero-length, end with ".crc" or
			// ".mtd". Create args list replace #FILE#, and invoke SystemML

			FileSystem inFS = FileSystemFactory.INSTANCE
					.getFileSystemFor(inDirPath);

			for (FileStatus fstat : inFS.listStatus(inDirPath)) {

				String fname = fstat.getPath().getName();

				if ((false == fstat.isDir()) && ('.' != fname.charAt(0))
						&& (fstat.getLen() > 0L) && (!fname.endsWith(".crc"))
						&& (!fname.endsWith(".mtd"))) {

					args = createArgsList(inDirPath, outDirPath, getVars(),
							fname, null);

					logStatusUpdate(makeStatus("DMLSCriptCmd.main "
							+ args.toString()));
				}
			}
			break;
		}

		case univariate: {

			// Iterate over all the files in input directory: #FILE#. Skip files
			// that start with a ".", are zero-length, end with ".crc" or
			// ".mtd". Look up the metadata file to retrieve the "kind" of
			// attribute, and invoke the assigned script with args list
			// accordingly.

			Map<String, JsonValue> swCase;
			String fname;

			// retrieve switch field from job definition
			JsonString swfld = (JsonString) ((JsonRecord) getVariable(DML_SWITCH_VAR))
					.getRequired(DML_SWITCH_FIELD_VAR);

			// retrieve switch cases from job definition
			JsonRecord swcases = (JsonRecord) ((JsonRecord) getVariable(DML_SWITCH_VAR))
					.getRequired(DML_SWITCH_CASES_VAR);

			FileSystem inFS = FileSystemFactory.INSTANCE
					.getFileSystemFor(inDirPath);

			for (FileStatus fstat : inFS.listStatus(inDirPath)) {

				fname = fstat.getPath().getName();

				if ((false == fstat.isDir()) && ('.' != fname.charAt(0))
						&& (fstat.getLen() > 0L) && (!fname.endsWith(".crc"))
						&& (!fname.endsWith(".mtd"))) {

					// retrieve switch field value from meta data file.
					JsonRecord mtd = (JsonRecord) JSONUtils
							.fileToJSON(new Path(inDirPath, fname + ".mtd"));
					JsonString fldValue = (JsonString) ((JsonRecord) mtd
							.getRequired(MTD_DESCR)).getRequired(swfld);

					// pick switch case using switch field value
					swCase = JMapToSMap(((JsonRecord) swcases
							.getRequired(fldValue)).asMap());

					// construct args list for switch case
					args = createArgsList(inDirPath, outDirPath, swCase, fname,
							null);

					// invoke SystemML
					logStatusUpdate(makeStatus("DMLSCriptCmd.main "
							+ args.toString()));
				}
			}
			break;
		}

		case bivariate: {

			// Iterate over all the files in input directory and pair them to
			// #FILE1#, #FILE2#. Skip files that start with a ".", are
			// zero-length, end with ".crc" or ".mtd". Look up the metadata
			// files for both to retrieve the "kind"s of attributes, pair them,
			// and invoke the assigned script with args
			// list accordingly.

			Map<String, JsonValue> swCase;
			String fname;
			TreeMap<String, String> fileList = new TreeMap<String, String>();

			// retrieve switch field from job definition
			JsonString swfld = (JsonString) ((JsonRecord) getVariable(DML_SWITCH_VAR))
					.getRequired(DML_SWITCH_FIELD_VAR);

			// retrieve switch cases from job definition
			JsonRecord swcases = (JsonRecord) ((JsonRecord) getVariable(DML_SWITCH_VAR))
					.getRequired(DML_SWITCH_CASES_VAR);

			FileSystem inFS = FileSystemFactory.INSTANCE
					.getFileSystemFor(inDirPath);

			// Retrieve file list
			for (FileStatus fstat : inFS.listStatus(inDirPath)) {

				fname = fstat.getPath().getName();

				if ((false == fstat.isDir()) && ('.' != fname.charAt(0))
						&& (fstat.getLen() > 0L) && (!fname.endsWith(".crc"))
						&& (!fname.endsWith(".mtd"))) {

					// retrieve switch field value from meta data file.
					JsonRecord mtd = (JsonRecord) JSONUtils
							.fileToJSON(new Path(inDirPath, fname + ".mtd"));
					JsonString fldValue = (JsonString) ((JsonRecord) mtd
							.getRequired(MTD_DESCR)).getRequired(swfld);

					// add to list
					fileList.put(fname, fldValue.toString());
				}
			}

			// Build list pairs and iterate
			for (Entry<String, String> f1 : fileList.entrySet()) {
				for (Entry<String, String> f2 : fileList.entrySet()) {

					if (f1.equals(f2)) {
						// nothing to do
					} else {

						// get switch field value
						JsonString swValue = new JsonString(f1.getValue()
								.concat(f2.getValue()));

						// get switch case
						JsonValue jswCase = swcases.get(swValue);
						if (jswCase == null) {
							// nothing to do as no script specified
						} else {

							swCase = JMapToSMap(((JsonRecord) jswCase).asMap());

							// construct args list for switch case
							args = createArgsList(inDirPath, outDirPath,
									swCase, f1.getKey(), f2.getKey());

							// invoke SystemML
							logStatusUpdate(makeStatus("DMLSCriptCmd.main "
									+ args.toString()));
						}
					}
				}
			}

			break;
		}

		}

		logStatusUpdate(makeStatus("End MT DmlJob Run."));
	}

	/*
	 * UTILITY METHODS
	 */

	/**
	 * Utility method to create a status object that marks this job instance as
	 * running, with the indicated message in an auxiliary field called "msg".
	 */
	protected static JobStatus makeStatus(String format, Object... args) {
		JobStatus ret = new JobStatus(JobState.RUNNING);
		ret.setProperty("msg", String.format(format, args));
		return ret;
	}
}
