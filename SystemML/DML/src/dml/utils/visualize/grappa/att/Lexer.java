/*
 *  This software may only be used by you under license from AT&T Corp.
 *  ("AT&T").  A copy of AT&T's Source Code Agreement is available at
 *  AT&T's Internet website having the URL:
 *  <http://www.research.att.com/sw/tools/graphviz/license/source.html>
 *  If you received this software without first entering into a license
 *  with AT&T, you have an infringing copy of this software and cannot use
 *  it without violating AT&T's intellectual property rights.
 */

package dml.utils.visualize.grappa.att;

import dml.utils.visualize.grappa.java_cup.runtime.Symbol;

import java.util.Hashtable;
import java.io.*;

/**
 * A class for doing lexical analysis of <i>dot</i> formatted input.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class Lexer
{
    /**
     * First character of lookahead.
     * Set to '\n' initially (needed for initial 2 calls to advance())
     */
    private int next_char = '\n';

    /**
     * Second character of lookahead.
     * Set to '\n' initially (needed for initial 2 calls to advance())
     */
    private int next_char2 = '\n';

    /**
     * Current line number for use in error messages.
     * Set to -1 to account for next_char/next_char2 initialization
     */
    private int current_line = -1;

    /**
     * Character position in current line.
     */
    private int current_position = 1;

    /**
     * EOF constant.
     */
    private static final int EOF_CHAR = -1;

    /**
     * needed to handle anonymous subgraphs since parser has no precedence
     */
    private boolean haveId = false;

    /**
     * needed for retreating
     */
    private int old_char;
    private int old_position;
    boolean retreated = false;

    /**
     * Count of total errors detected so far.
     */
    private int error_count = 0;

    /**
     *  Count of warnings issued so far
     */
    private int warning_count = 0;

    /**
     *  hash tables to hold symbols
     */
    private Hashtable keywords = new Hashtable(32);
    private Hashtable char_symbols = new Hashtable(32);

    private Reader inReader;
    private PrintWriter errWriter = null;

    /**
     * common StringBuffer (suggested by Ginny Travers (bbn.com))
     */
    private StringBuffer cmnstrbuf = new StringBuffer();


    /**
     * Create an instance of <code>Lexer</code> that reads from <code>input</code> and
     * sends error messages to <code>error</code>.
     *
     * @param input input <code>Reader</code> object
     * @param error error output <code>Writer</code> object
     *
     * @exception IllegalArgumentException whenever <code>input</code> is null
     */
    public Lexer(Reader input, PrintWriter error) throws IllegalArgumentException {
	super();
	if (input == null) {
	    throw new IllegalArgumentException("Reader cannot be null");
	}
	inReader = input;
	errWriter = error;
    }

    /**
     * Initialize internal tables and read two characters of input for
     * look-ahead purposes.
     *
     * @exception IOException if <code>advance()</code> does
     * @see Lexer#advance()
     */
    public void init() throws IOException {
	// set up the keyword table
	keywords.put("strict", new Integer(Symbols.STRICT));
	keywords.put("strictdigraph", new Integer(Symbols.STRICTDIGRAPH));
	keywords.put("strictgraph", new Integer(Symbols.STRICTGRAPH));
	keywords.put("digraph", new Integer(Symbols.DIGRAPH));
	keywords.put("graph", new Integer(Symbols.GRAPH));
	keywords.put("subgraph", new Integer(Symbols.SUBGRAPH));
	keywords.put("node", new Integer(Symbols.NODE));
	keywords.put("edge", new Integer(Symbols.EDGE));
	keywords.put("--", new Integer(Symbols.ND_EDGE_OP));
	keywords.put("->", new Integer(Symbols.D_EDGE_OP));

	// set up the table of single character symbols
	char_symbols.put(new Integer(';'), new Integer(Symbols.SEMI));
	char_symbols.put(new Integer(','), new Integer(Symbols.COMMA));
	char_symbols.put(new Integer('{'), new Integer(Symbols.LCUR));
	char_symbols.put(new Integer('}'), new Integer(Symbols.RCUR));
	char_symbols.put(new Integer('['), new Integer(Symbols.LBR));
	char_symbols.put(new Integer(']'), new Integer(Symbols.RBR));
	char_symbols.put(new Integer('='), new Integer(Symbols.EQUAL));
	char_symbols.put(new Integer(':'), new Integer(Symbols.COLON));

	// read two characters of lookahead
	advance();
	advance();
    }
    /**
     * Advance the scanner one character in the input stream.  This moves
     * next_char2 to next_char and then reads a new next_char2.
     *
     * @exception IOException whenever a problem reading from <code>input</code> is encountered
     */
    public void advance() throws IOException {
	if(retreated) {
	    retreated = false;
	    int tmp_char = old_char;
	    old_char = next_char;
	    next_char = next_char2;
	    next_char2 = tmp_char;
	} else {
	    old_char = next_char;
	    next_char = next_char2;
	    if (next_char == EOF_CHAR) {
		next_char2 = EOF_CHAR;
	    } else {
		next_char2 = inReader.read();
	    }
	}

	/*
	 * want to ignore a new-line if preceeding character is a backslash
	 */
	if (next_char == '\\' && (next_char2 == '\n' || next_char2 == '\r')) {
	    next_char = next_char2;
	    next_char2 = inReader.read();
	    if(next_char == '\r' && next_char2 == '\n') {
		next_char = next_char2;
		next_char2 = inReader.read();
	    }
	    next_char = next_char2;
	    next_char2 = inReader.read();
	}


	/*
	 * want to treat '\r' or '\n' or '\r''\n' as end-of-line,
	 * but in all cases return only '\n'
	 */
	if(next_char == '\r') {
	    if(next_char2 == '\n') {
		next_char2 = inReader.read();
	    }
	    next_char = '\n';
	}
	// count this
	if (old_char == '\n') {
	    current_line++;
	    old_position = current_position;
	    current_position = 1;
	} else {
	    current_position++;
	}
    }

    private void retreat() {
	if(retreated) return;
	retreated = true;
	if(old_char == '\n') {
	    current_line--;
	    current_position = old_position;
	} else {
	    current_position--;
	}
	int tmp_char = next_char2;
	next_char2 = next_char;
	next_char = old_char;
	old_char = tmp_char;
    }

    /**
     * Emit an error message.  The message will be marked with both the
     * current line number and the position in the line.  Error messages
     * are printed on print stream passed to Lexer (if any) and a
     * GraphParserException is thrown.
     *
     * @param message the message to print.
     */
    private void emit_error(String message) {
	String output = "Lexer" + getLocation() + ": " + message;
	if(errWriter != null) {
	    errWriter.println("ERROR: " + output);
	}
	error_count++;
	throw new GraphParserException(output);
    }

    /**
     * Get the current location in the form "[line_number(character_offser)]".
     * 
     * @return info about the current position in the input
     */
    public String getLocation() {
	return "[" + current_line + "(" + current_position + ")]";
    }
  

    /**
     * Emit a warning message.  The message will be marked with both the
     * current line number and the position in the line.  Messages are
     * printed on print stream passed to Lexer (if any).
     *
     * @param message the message to print.
     */
    private void emit_warn(String message) {
	if(errWriter != null) {
	    errWriter.println("WARNING: Lexer" + getLocation() + ": " + message);
	}
	warning_count++;
    }

    /**
     * Check if character is a valid id character;
     * @param ch the character in question.
     */
    public static boolean id_char(int ch) {
	return(Lexer.id_char((char)ch));
    }

    /**
     * Check if character is a valid id character;
     * @param ch the character in question.
     */
    public static boolean id_char(char ch) {
	return((Character.isJavaIdentifierStart(ch) && Character.getType(ch) != Character.CURRENCY_SYMBOL) || Character.isDigit(ch));
    }

    /**
     * Try to look up a single character symbol, returns -1 for not found.
     * @param ch the character in question.
     */
    private int find_single_char(int ch) {
	Integer result;

	result = (Integer) char_symbols.get(new Integer((char) ch));
	if (result == null) {
	    return -1;
	} else {
	    return result.intValue();
	}
    }

    /**
     * Handle swallowing up a comment.  Both old style C and new style C++
     * comments are handled.
     */
    private void swallow_comment() throws IOException {
	// next_char == '/' at this point.

	// Is it a traditional comment?
	if (next_char2 == '*') {
	    // swallow the opener
	    advance();
	    advance();

	    // swallow the comment until end of comment or EOF
	    for (;;) {
		// if its EOF we have an error
		if (next_char == EOF_CHAR) {
		    emit_error("Specification file ends inside a comment");
		    return;
		}
		// if we can see the closer we are done
		if (next_char == '*' && next_char2 == '/') {
		    advance();
		    advance();
		    return;
		}
		// otherwise swallow char and move on
		advance();
	    }
	}
	// is its a new style comment
	if (next_char2 == '/') {

	    // swallow the opener
	    advance();
	    advance();

	    // swallow to '\n', '\f', or EOF
	    while (next_char != '\n' && next_char != '\f' && next_char != EOF_CHAR) {
		advance();
	    }

	    return;

	}
	// shouldn't get here, but... if we get here we have an error
	emit_error("Malformed comment in specification -- ignored");
	advance();
    }

    /**
     * Swallow up a quote string.  Quote strings begin with a double quote
     * and include all characters up to the first occurrence of another double
     * quote (there is no way to include a double quote inside a quote string).
     * The routine returns a Symbol object suitable for return by the scanner.
     */
    private Symbol do_quote_string() throws IOException {
	String result_str;

	// at this point we have lookahead of a double quote -- swallow that
	advance();

	synchronized(cmnstrbuf) {
	    cmnstrbuf.delete(0,cmnstrbuf.length()); // faster than cmnstrbuf.setLength(0)!
	    // save chars until we see a double quote
	    while (!(next_char == '"')) {
		// skip line break
		if (next_char == '\\' && next_char2 == '"') {
		    advance();
		}
		// if we have run off the end issue a message and break out of loop
		if (next_char == EOF_CHAR) {
		    emit_error("Specification file ends inside a code string");
		    break;
		}
		// otherwise record the char and move on
		cmnstrbuf.append(new Character((char) next_char));
		advance();
	    }

	    result_str = cmnstrbuf.toString();
	}

	// advance past the closing double quote and build a return Symbol
	advance();
	haveId = true;
	return new Symbol(Symbols.ATOM, result_str);
    }

    /**
     * Swallow up an html-like string.  Html-like strings begin with a '<'
     * and include all characters up to the first matching occurrence of a '>'
     * The routine returns a Symbol object suitable for return by the scanner.
     */
    private Symbol do_html_string() throws IOException {
	String result_str;
	int angles = 0;

	synchronized(cmnstrbuf) {
	    cmnstrbuf.delete(0,cmnstrbuf.length()); // faster than cmnstrbuf.setLength(0)!
	    // save chars until we see a double quote
	    do {
		if (next_char == EOF_CHAR) {
		    emit_error("Specification file ends inside an html string");
		    break;
		}
		    
		if (next_char == '<')
		    angles++;
		else if (next_char == '>')
		    angles--;

		cmnstrbuf.append(new Character((char) next_char));
		advance();
	    } while(angles > 0);

	    result_str = cmnstrbuf.toString();
	}

	// advance past the closing double quote and build a return Symbol
	advance();
	haveId = true;
	return new Symbol(Symbols.ATOM, result_str);
    }

    /**
     * Process an identifier.  Identifiers begin with a letter, underscore,
     * or dollar sign, which is followed by zero or more letters, numbers,
     * underscores or dollar signs.  This routine returns an Symbol suitable
     * for return by the scanner.
     */
    private Symbol do_id() throws IOException {
	String result_str;
	Integer keyword_num;
	char buffer[] = new char[1];

	// next_char holds first character of id
	buffer[0] = (char) next_char;

	synchronized(cmnstrbuf) {
	    cmnstrbuf.delete(0,cmnstrbuf.length()); // faster than cmnstrbuf.setLength(0)!
	    cmnstrbuf.append(buffer, 0, 1);
	    advance();

	    // collect up characters while they fit in id
	    while (id_char(next_char)) {
		buffer[0] = (char) next_char;
		cmnstrbuf.append(buffer, 0, 1);
		advance();
	    }
	    // extract a string and try to look it up as a keyword
	    result_str = cmnstrbuf.toString();
	}

	keyword_num = (Integer) keywords.get(result_str);

	// if we found something, return that keyword
	if (keyword_num != null) {
	    haveId = false;
	    return new Symbol(keyword_num.intValue());
	}

	// otherwise build and return an id Symbol with an attached string
	haveId = true;
	return new Symbol(Symbols.ATOM, result_str);
    }

    /**
     * The actual routine to return one Symbol.  This is normally called from
     * next_token(), but for debugging purposes can be called indirectly from
     * debug_next_token().
     */
    private Symbol real_next_token() throws IOException {
	int sym_num;

	for (;;) {
	    // look for white space
	    if (next_char == ' ' || next_char == '\t' || next_char == '\n' ||
		next_char == '\f') {

		// advance past it and try the next character
		advance();
		continue;
	    }

	    // look for edge operator
	    if (next_char == '-') {
		if (next_char2 == '>') {
		    advance();
		    advance();
		    haveId = false;
		    return new Symbol(Symbols.D_EDGE_OP);
		} else if (next_char2 == '-') {
		    advance();
		    advance();
		    haveId = false;
		    return new Symbol(Symbols.ND_EDGE_OP);
		}
	    }

	    // look for a single character symbol
	    sym_num = find_single_char(next_char);
	    if (sym_num != -1) {
		if (sym_num == Symbols.LCUR && !haveId) {
		    Symbol result = new Symbol(Symbols.SUBGRAPH);
		    haveId = true;
		    retreat();
		    return result;
		}

		// found one -- advance past it and return a Symbol for it
		advance();
		haveId = false;
		return new Symbol(sym_num);
	    }

	    // look for quoted string
	    if (next_char == '"') {
		return do_quote_string();
	    }

	    // look for html-like string
	    if (next_char == '<') {
		return do_html_string();
	    }

	    // look for a comment
	    if (next_char == '/' && (next_char2 == '*' || next_char2 == '/')) {
		// swallow then continue the scan
		swallow_comment();
		continue;
	    }

	    // look for an id or keyword
	    if (id_char(next_char)) {
		return do_id();
	    }

	    // look for EOF
	    if (next_char == EOF_CHAR) {
		haveId = false;
		return new Symbol(Symbols.EOF);
	    }

	    // if we get here, we have an unrecognized character
	    emit_warn("Unrecognized character '" +
		      new Character((char) next_char) + "'(" + next_char +
		      ") -- ignored");

	    // advance past it
	    advance();
	}
    }

    /**
     * Return one Symbol.  This method is the main external interface to
     * the scanner.
     * It consumes sufficient characters to determine the next input Symbol
     * and returns it.
     *
     * @exception IOException if <code>advance()</code> does
     */
    public Symbol next_token(int debugLevel) throws IOException {
	if(debugLevel > 0) {
	    Symbol result = real_next_token();
	    if(errWriter != null && debugLevel >= 5) {
		errWriter.println("DEBUG: Lexer: next_token() => " + result.sym);
	    }
	    return result;
	} else {
	    return real_next_token();
	}
    }
}
