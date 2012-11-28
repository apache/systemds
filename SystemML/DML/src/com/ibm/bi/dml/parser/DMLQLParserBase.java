package com.ibm.bi.dml.parser;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;



 
 
/**
 * Base class for the DMLQL parser. We put functionality into this class so that
 * we don't have to edit the java in the JavaCC file.
 */
public abstract class DMLQLParserBase {
	protected static final Log LOG = LogFactory.getLog(DMLQLParserBase.class.getName());
	
	/** File encoding to use if no other encoding is explicitly specified. */
	protected static final String DEFAULT_ENCODING = "UTF-8";

	/** Encoding of the file currently being parsed. */
	private String fileEncoding = DEFAULT_ENCODING;

	/**
	 * Hook for use by generated Java code, since the generated constructor will
	 * expect the base class's constructor not to take any arguments.
	 * 
	 * @param fileEncoding
	 *            encoding of the file to be parsed
	 */
	public void setFileEncoding(String fileEncoding) {
		this.fileEncoding = fileEncoding;
	}

	/**
	 * Generate a ParseException with a token pointer and a message.
	 * 
	 * We put this method here because we can't add a new constructor to the
	 * generated class ParseException.
	 */
	public static ParseException makeException(String msg, Token token) {
		ParseException ret = new ParseException(msg);
		ret.currentToken = token;
		return ret;
	}

	/**
	 * Generate a ParseException with a token pointer and a message.
	 * 
	 * We put this method here because we can't add a new constructor to the
	 * generated class ParseException.
	 * 
	 * @param token
	 *            token indicating the location of the problem
	 * @param format
	 *            format string, as in {@link String#format(String, Object...)}
	 * @param args
	 *            arguments for format string
	 */
	public static ParseException makeException(Token token, String format,
			Object... args) {
		String msg = String.format(format, args);
		ParseException ret = new ParseException(msg);
		ret.currentToken = token;
		return ret;
	}




	/**
	 * Default constructor (the only kind of constructor we can support, due to
	 * JavaCC constraints; the JavaCC-generated class will have a default
	 * constructor that expects the superclass argument to take no arguments).
	 * Performs initialization of internal data structures for the base class.
	 */
	protected DMLQLParserBase() {

	}




	/**
	 * Remove quotes and escaped quotes from a quoted string returned by the
	 * lexer.
	 * 
	 * @param quotechar
	 *            character (e.g. " or /) used to indicate start/end quotation
	 * @param string
	 *            string to be dequoted
	 * @return string with quotes removed
	 * @throws ParseException
	 */
	protected static String dequoteStr(char quotechar, Token quotedString)
	throws ParseException {
		try {

			String ret = dequoteStr(quotechar, quotedString.image);

			return ret;
		} catch (IllegalArgumentException e) {
			throw makeException(e.getMessage(), quotedString);
		}
	}


	/**
	 * Removes the quotes from a quoted string as returned by a parser.
	 * 
	 * @param quotechar
	 * @param str
	 *            input string, surrounded by quotes
	 * @return the string, with quotes removed and any escaped quotes inside the
	 *         string de-escaped
	 */
	public static final String dequoteStr(char quotechar, String str) {
		if (str.charAt(0) != quotechar
				|| str.charAt(str.length() - 1) != quotechar) {
			LOG.error("Can't dequote string '" + str + "'");
			throw new IllegalArgumentException("Can't dequote string '" + str + "'");
		}

		StringBuilder sb = new StringBuilder();

		final char ESCAPE = '\\';

		// Walk through the string from start to end, removing escapes.
		for (int pos = 1; pos < str.length() - 1; pos++) {
			if (str.charAt(pos) == ESCAPE && str.charAt(pos + 1) == quotechar) {
				// When we find ESCAPE followed by the quote char, skip the
				// escape; the quote character will be passed through.
				if (str.length() - 1 == pos) {
					LOG.error("Escape character at end of string");
					throw new IllegalArgumentException("Escape character at end of string");
				}
			} else {
				// All other characters just get passed through.
				sb.append(str.charAt(pos));
			}
		}

		return sb.toString();
	}
	/**
	 * Generated code should call this method at the end of parsing, right
	 * before returning the completed parse tree.
	 */
	protected void tearDown() throws ParseException {

	}

	/** Placeholder for the main parser entry point in generated code. */
	public abstract DMLProgram __inputInternal() throws ParseException;

	/**
	 * Wrapper around the main parser entry point; adds file names and line
	 * numbers to thrown exceptions, and handles the input file stack.
	 * 
	 * No return value; parse tree nodes go into the parser's catalog.
	 */
	public DMLProgram parse() throws ParseException {

		try {
			return __inputInternal();
		} catch (ParseException e) {
			LOG.error("Failed in DMLProgram parsing: " + e.getMessage());
			return null;
		} catch (TokenMgrError e) {
			LOG.error("Error: " + e.getMessage());
		}
		tearDown();
		return null;
	}

	



}
