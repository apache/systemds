package com.ibm.bi.dml.utils;

import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class to help setting variables in a script. 
 */
public class ParameterBuilder {
	/** String used to replace variables in scripts */
	private static String _RS = "\\$\\$";

	/**
	 * <p>
	 * Replaces variables in a DML or R script with the specified values. A
	 * variable of format $$name$$ will be replaced where the name is used to
	 * identify the variable in the hashmap containing the belonging value.
	 * </p>
	 * 
	 * @param strScriptPathName
	 *            filename of the DML script
	 * @param variables
	 *            hashmap containing all the variables and their replacements
	 */
	public static void setVariablesInScript(String strScriptPathName, HashMap<String, String> variables) {
		try {
			String strScript = strScriptPathName;
			String strTmpScript = strScript + "t";

			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(strScript)));
			FileOutputStream out = new FileOutputStream(strTmpScript);
			PrintWriter pw = new PrintWriter(out);
			String content;
			Pattern unresolvedVars = Pattern.compile(_RS + ".*" + _RS);
			/**
			 * sothat variables, which were not assigned, are replaced by an
			 * empty string
			 */
			while ((content = in.readLine()) != null) {
				for (String variable : variables.keySet()) {
					Pattern pattern = Pattern.compile(_RS + variable + _RS);
					Matcher matcher = pattern.matcher(content);
					while (matcher.find()) {
						content = content.replaceFirst(matcher.group().replace("$", "\\$"), variables.get(variable));
					}
				}
				Matcher matcher = unresolvedVars.matcher(content);
				content = matcher.replaceAll("");
				pw.println(content);
			}
			pw.close();
			out.close();
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to set variables in dml script: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Replaces variables in a DML or R script with the specified values. A
	 * variable of format $$name$$ will be replaced where the name is used to
	 * identify the variable in the hashmap containing the belonging value.
	 * </p>
	 * 
	 * @param strScriptDirectory
	 *            directory which contains the DML script
	 * @param strScriptFile
	 *            filename of the DML script
	 * @param variables
	 *            hashmap containing all the variables and their replacements
	 */
	public static void setVariablesInScript(String strScriptDirectory, String strScriptFile,
			HashMap<String, String> variables) {

		String strScript = strScriptDirectory + strScriptFile;
		setVariablesInScript(strScript, variables);
	}
}
