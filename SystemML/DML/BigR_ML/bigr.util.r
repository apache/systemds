# bigr.util
#
# This file contains utility functions that can be used all
# accross the project

# A method to easily concatenate two strings
#' @keywords internal
"%++%" <- function(x, y) {
    paste(x, y, sep = "")
}

# An environment to store BigR variables
cat("Attaching...")
bigr.env <- new.env()

with(bigr.env, {
# This package's name
    PACKAGE_NAME <- "bigr";
    
# LOG LEVEL constants
    BIGR_HOME <- Sys.getenv("BIGR_HOME");
    INFO_LOG_LEVEL <- 4;
    WARN_LOG_LEVEL <- 2;
    ERR_LOG_LEVEL <- 1;
    NO_LOG_LEVEL <- 0;
    DEFAULT_LOG_LEVEL <- ERR_LOG_LEVEL + WARN_LOG_LEVEL;
    LOG_LEVEL <- DEFAULT_LOG_LEVEL;
    
# bigr.connection constants
    
    # Bigr path
    BIGR_HOME <- Sys.getenv("BIGR_HOME");
    
    # JaQL module paths
    JAQL_MODULES <- c("BigR.jaql",
                      "DataManipulation.jaql",
                      "Analytics.jaql",
                      "Apply.jaql",
                      "Sampling.jaql",
                      "ML.jaql");
    
    # Default database to be used if none is specified
    DEFAULT_DATABASE <- "default";
    
    # The bigr.connection class name
    CONNECTION_CLASS_NAME <- "bigr.connection";
    
    # The maximum number of rows to be returned by a JaQL query
    DEFAULT_ROW_LIMIT <- 1000;
    
# bigr.dataset constants
    DATASET_CLASS_NAME <- "bigr.dataset";
    
    # Possible data sources
    TEXT_FILE <- "TEXT";
    JSON_FILE <- "TEXT";
    LINE_FILE <- "LINE"
    BIG_SQL <- "BIGSQL";
    TRANSFORM <- "TRANSF"; # When a bigr.dataset is the result of a transformation (e.g., projection or filtering)
    
    EMPTY_TABLE_EXPRESSION <- "[[]]";
    
    # A list of supported data formats
    DATA_SOURCES <- c(TEXT_FILE, BIG_SQL, TRANSFORM, LINE_FILE);
    
    # Supported data types
    DATA_TYPES <- c(character="string", 
                    numeric="double",
                    numeric="float", 
                    integer="long", 
                    integer="tinyint", 
                    integer="smallint", 
                    integer="int", 
                    integer="bigint",
                    logical="boolean", 
                    factor="string");
    
    SUPPORTED_DATA_TYPES <- c("character", "numeric", "integer", "logical");
    
    # The number of rows returned by head and tail methods
    DEFAULT_HEAD_ROWS <- 6;
    
    # The default delimiter for text files
    DEFAULT_DELIMITER <- ",";
    
    # NA string
    DEFAULT_NA_STRING <- "";
    
    # Local processing
    DEFAULT_LOCAL_PROCESSING <- TRUE;
    
    # If no rows are returned by a query
    EMPTY_DATA <- data.frame();

    # A global bigr.connection object
    CONNECTION <- NULL;
    
    COUNT_CACHE <- list()
        
# bigr.vector constants
    
    # The bigr.vector class name
    VECTOR_CLASS_NAME <- "bigr.vector";

# bigr.frame constants

    # The bigr.frame class name
    FRAME_CLASS_NAME <- "bigr.frame";
    
    # The default column names if none specified
    DEFAULT_COLNAMES <- c("x1");
    
    # The default column types if none specified
    DEFAULT_COLTYPES <- c("character");
    
    DEFAULT_NBINS = 10;
    
    # Aggregate functions for numeric columns
    ALL_AGGREGATE_FUNCTIONS = c("count", "countNA", "countnonNA", "min", "max", "sum", "avg", "mean", "sd", "var");
    
    ALL_NOMINAL_AGGREGATE_FUNCTIONS = c("count", "countNA", "countnonNA", "min", "max");    

    # Default aggregate functions for numeric columns
    DEFAULT_NUMERIC_AGGREGATE_FUNCTIONS = c("count", "min", "max", "sum", "mean");
    
    # Aggregate functions for nominal columns
    DEFAULT_NOMINAL_AGGREGATE_FUNCTIONS = c("count", "min", "max");
    
    # The list of aggregate functions which always return a numeric value
    NUMERIC_TYPE_AGGREGATE_FUNCTIONS = c("count", "sum", "mean", "countNA", "countnonNA");
    
# bigr.function constants
    
    FUNCTION_CLASS_NAME <- "bigr.function";
    
    REGISTERED_FUNCTIONS <- list();
 
    
# bigr.sampling constants
    RANDOM_SEED <- 71

    # The bigr.list class name
    LIST_CLASS_NAME <- "bigr.list";
    
    # warnings and errors log file
    LOGDIR <- "";
    LOGFILE <- "";
    LOGCOLTYPE <- c();
    LOGCOLNAME <- c();
    
    # JDBC class name
    JDBC_CLASS_NAME <- "bigr.jdbc";
    
    # Default number of rows to fetch in one trip
    JDBC_FETCH_SIZE <- 16384;

    # Random numbers / RNG support
    DEFAULT_RNG <- "bigr.default.rng"
}) # End with


# Checks whether a given object is NULL or empty.
#
# This function returns true if the given object is NULL, NA, "". Also, if
# it is an array which only contains NULL, NA or "" values.
# @param x an object or variable to be evaluated
# @return a logical value indicating whether the given object is NULL, NA or empty
# @rdname internal.bigr.isNullOrEmpty
# @keywords internal
.bigr.isNullOrEmpty <- function(x) {
    logSource <- ".bigr.isNullOrEmpty"
    if (missing(x)) {
        bigr.err(logSource, 
                 sprintf("A value for parameter '%s' must be specified.", deparse(substitute(x))))
    }
    if (class(x) == bigr.env$FRAME_CLASS_NAME | class(x) == bigr.env$VECTOR_CLASS_NAME
        | class(x) == bigr.env$DATASET_CLASS_NAME) {
        return(FALSE)
    }    
    if (is.null(x)) {
        return(TRUE)  
    } else if (length(x) < 1) {
        return(TRUE)
    }
    if (class(x) == "character" | class(x) == "numeric" | class(x) == "logical" | class(x) == "integer") {
        if (all(is.na(x) | is.null(x) | x == "") ) {
            return(TRUE)
        }
    }
    return(FALSE)
}

.bigr.hasNullOrEmpty <- function(x) {
    logSource <- ".bigr.isNullOrEmpty"
    if (.bigr.isNullOrEmpty(x)) {
        return(TRUE)
    }
    for (i in 1:length(x)) {
        if (.bigr.isNullOrEmpty(x[i])) {
            return(TRUE)
        }
    }    
    return(FALSE)
}

# Checks if two vectors are equals
.bigr.equals <- function(a, b) {
    logSource <- ".bigr.equals"
    if (!is.vector(a) | !is.vector(b)) {
        bigr.err(logSource, "Cannot apply equals function to non-vector operands")
    }
    if (is.null(a) | is.null(b)) {
        return(FALSE)
    }
    if (length(a) != length(b)) {
        return(FALSE)
    }
    if (all(a == b)) {
        return(TRUE)
    } else {
        return(FALSE)
    }
}

# Indicates whether a string pattern is contained into a given string x
.bigr.contains <- function(x, pattern) {
    logSource <- "contains"
    if (!.bigr.isNullOrEmpty(x) & !.bigr.isNullOrEmpty(pattern)) {
        if (class(x) != "character" | class(pattern) != "character") {
            bigr.err(logSource, "Contains cannot be applied to " %++% class(x) %++% ", " %++% class(pattern))
        }
        return(regexec(pattern, x, fixed=TRUE)[[1]][1] != -1)    
    } else {
        return(FALSE)
    }
}

# Maps a JaQL data type to the corresponding R type
.bigr.getJaqlType <- function(rType) {
    if (.bigr.isNullOrEmpty(rType) | class(rType) != "character") {
        return(NULL)
    }
    if (rType %in% names(bigr.env$DATA_TYPES)) {
        return(bigr.env$DATA_TYPES[[rType]])
    } else {
        return(NULL)
    }
}

# Maps an R data type to the corresponding JaQL type
.bigr.getRType <- function(jaqlType) {
    if (is.null(jaqlType) | class(jaqlType) != "character") {
        return(NULL)
    }
    if (jaqlType %in% bigr.env$DATA_TYPES) {    
        pos = match(jaqlType, bigr.env$DATA_TYPES)
        return(names(bigr.env$DATA_TYPES)[pos])
    } else {
        return(NULL)
    }
}

#' Given a vector of \code{character}, appends the corresponding number of spaces at the
#' end of each element, in order to make them all have the same length.
#'
#' @param x an array of strings to be aligned
#' @return an array of aligned strings
#' @rdname internal.bigr.align.strings
#' @keywords internal
.bigr.align.strings <- function(x) {    
    maxLength <- -1
    for (i in 1:length(x)) {
        newLength <- nchar(x[i]) 
        if (newLength > maxLength) {
            maxLength <- newLength
        }
    }    
    for (i in 1:length(x)) {
        spaces <- paste(rep(" ", maxLength - nchar(x[i])), sep="", collapse="")
        x[i] <- x[i] %++% spaces
    }
    return(x)    
}


#' Checks whether a given value is an integer number
#'
#' @param x the number to be checked
#' @return \code{TRUE} if \code{x} is an integer. \code{FALSE} otherwise.
#' @rdname internal.bigr.is.integer
#' @keywords internal
.bigr.is.integer <- function(x) {
    if (.bigr.isNullOrEmpty(x)) {
        return(FALSE)
    }
    if (length(x) != 1) {
        return(FALSE)
    }
    if (!is.numeric(x)) {
        return(FALSE)
    }
    if (x != as.integer(x)) {
        return(FALSE)
    }
    return(TRUE)
}
#' Changes the maximum number of rows/elements to be displayed by BigR.
#'
#' @param rowLimit the maximum number of rows/elements to be displayed
bigr.setRowLimit <- function(rowLimit) {
    logSource <- "bigr.setRowLimit"
    if (!.bigr.is.integer(rowLimit)) {
        bigr.err(logSource, "Row limit must be a positive integer value.")
    }
    if (rowLimit < 0) {
        bigr.err(logSource, "Row limit must be a positive integer value.")
    }
    options(bigr.row.limit = rowLimit)
    invisible(TRUE)
}

bigr.getRowLimit <- function() {
    getOption("bigr.row.limit")
}

# TODO: Check for copyright issues
#' Returns the number of days from the beginning of the Julian calendar.
#'
#' @keywords internal
.bigr.JDN <- function(month, day, year) {
    a <- floor((14 - month) / 12)
    y <- (year + 4800) - a
    m <- month + 12 * a - 3
    JDN <- day + floor((153 * m + 2) / 5) + 365 * y + floor(y / 4) - 32083
    return(JDN)
}

#' Returns the system time in milliseconds since January 1, 2000.
#'
#' @keywords internal
.bigr.currentTimeMillis <- function() {
    sysDate = as.character(Sys.time())
    year = as.integer(substring(sysDate, 1, 4))
    month = as.integer(substring(sysDate, 6, 7))
    day = as.integer(substring(sysDate, 9, 10))
    sysTime = format(Sys.time(), "%H:%M:%OS3")
    hour = as.integer(substring(sysTime, 1, 2))
    min = as.integer(substring(sysTime, 4, 5))
    sec = as.integer(substring(sysTime, 7, 8))
    millis = as.integer(substring(sysTime, 10, 12))
    
    # Get the number of days from January 1, 2000
    JDN <- .bigr.JDN(month, day, year) - 2451545
    JDN * 24 * 3600 * 1000 + hour * 3600 * 1000 + min * 60 * 1000 + sec * 1000 + millis
}

# This function returns the element count of a bigr.dataset if
# it has been cached. Otherwise, it returns NULL.
.bigr.getCachedCount <- function(dataset) {
    bigr.env$COUNT_CACHE[[dataset@tableExpression]]
}

.bigr.getShortType <- function(type) {
    if (type == "character") {
        "chr"
    } else if (type == "numeric") {
        "num"
    } else if (type == "logical") {
        "logi"
    } else if (type == "integer") {
        "int"
    } 
}

##################################################################################
# The following methods are used for the remote execution / JaQL to R invocation #
##################################################################################

#' Return the Jaql input schema for the bigr.frame
#'
#' @param data the bigr.frame object
#' @return JSON schema for the bigr.frame
#' @keywords internal
.bigr.frame.jaqlInputSchema <- function(data) {
    jaqlTypes <- sapply(data@coltypes, .bigr.getJaqlType)
    sc <- "schema {data: [ [" %++% paste(jaqlTypes, collapse=", ") %++% "]* ]}"
}

#' retrieve the data type of the bigr.vector object
#'
#' @param x  the bigr.vector object
#' @return the data type for the bigr.vector object
#' @keywords internal
bigr.Vector.getDataType <- function(x) {
    return (x@dataType)
}

#' Search for the JDBC driver path under CLASSPATH or bigr root directory
#' 
#' @return the JDBC driver jar file path for BigSQL server
#' @return jdbc driver path
#' @keywords internal
.bigr.getJDBCDriverPath <- function() {
    logSource <- ".bigr.getJDBCDriverPath"
    
    driverPath <- ""
    # search CLASSPATH first
    if ("Windows" == Sys.info()["sysname"]) {
        classpath <- unlist(strsplit(Sys.getenv("CLASSPATH"), ";"))
    } else {
        classpath <- unlist(strsplit(Sys.getenv("CLASSPATH"), ":"))
    }
    idx <- as.numeric(grep("bigsql-jdbc-driver", classpath))
    if (length(idx) > 0) {
        driverPath <- classpath[idx[[1]]]
        if (length(idx) > 1) {
            warning("Two or more jdbc drivers are found, " %++% driverPath %++% "is used. If you want to use other drivers, please pass the driverPath through bigr.connect")
        }
        
    } else {
        # search package root
        bigrRoot <- system.file(package=bigr.env$PACKAGE_NAME)
        bigrRootFiles <- list.files(bigrRoot)
        idx <- as.numeric(grep("bigsql-jdbc-driver", bigrRootFiles))
        if (length(idx) > 0) {
            driverPath <- bigrRoot %++% "/" %++% bigrRootFiles[idx[[1]]]
            if (length(idx) > 1) {
                bigr.warn(logSource, "Two or more jdbc drivers are found, " %++% driverPath %++% "is used. If you want to use other drivers, please pass the driverPath through bigr.connect")
            }
        }
    }
    
    if ("" == driverPath) {
        bigr.err(logSource, "Cannot find BigSQL JDBC Driver...")
    }
    return (driverPath)
}

#' Returns the data.frame that contains the log filename for each group or batch
#' @param x optional or the bigr.frame or bigr.list object that has a log summary file
#' @return all log filenames and status when the bigr.frame or bigr.list are genearted through *apply functions
bigr.getLogFiles <- function(x) {
    logSource <- "bigr.getLogFiles"
    logfile <- NULL
    if (missing(x)) {
        logdir <- bigr.env$LOGDIR
        logfile <- bigr.env$LOGFILE
        logcolnames <- bigr.env$LOGCOLNAME
        logcoltypes <- bigr.env$LOGCOLTYPE
    }
    else if ((bigr.env$LIST_CLASS_NAME == class(x)) || (bigr.env$FRAME_CLASS_NAME == class(x))) {
        if (0 == length(x@logInfo)) {
            bigr.err(logSource, "The object does not have any log files associated...")
        }
        logdir <- dirname(x@dataPath)
        logfile <- x@logInfo[[1]]
        logcolnames <- x@logInfo[[2]]
        logcoltypes <- x@logInfo[[3]]
    }
    
    if (is.null(logfile)) {
        bigr.err(logSource, "The object does not have any log files associated...")
    }
    else {
        p <- as.data.frame(new(bigr.env$FRAME_CLASS_NAME, bigr.env$TEXT_FILE, logfile, ",", logcolnames, logcoltypes, header=FALSE))
        n <- length(logcoltypes)
        p[,n] <- paste(logdir, "/", p[,n],sep="")
        p
    }
}

#' Returns all the log entries for a particular processing
#' @param x log filename
#' @return the logging entries in the log file
bigr.showLogFile <- function(x) {
    .bigr.execQuery("localRead(lines(location='" %++% x %++% "'))")
}

#' Update value for a configuration parameter in bigr.env
#' @param x the configuration parameter=value in character
#' @keywords internal
bigr.setConfig <- function(x) {
    logSource <- "bigr.setConfig"
    if (missing(x)) {
        print("JDBC_FETCH_SIZE = " %++% bigr.env$JDBC_FETCH_SIZE)
    }
    else {
        if (FALSE == grep("=", x)) {
            bigr.err(logSource, "please use the correct syntax")
        }
        else {
            config <- strsplit(x, "=")
            param <- config[[1]][[1]]
            val <- config[[1]][[2]]
            if (toupper(param) == "JDBC_FETCH_SIZE") {
                bigr.env$JDBC_FETCH_SIZE <- val
            }
        }
    }
}

# .bigr.checkParameter
#
# Utility function used to raise errors when invalid parameter values are specified.
.bigr.checkParameter <- function(logSource, parm, expected) {

    # Format the parameter name
    parmName <- as.character(substitute(parm))
    
    # Format the 'expected' values
    if (length(expected) > 1) {
        if (is.character(expected))
            expectedStr <- dQuote(expected)
        else
            expectedStr <- expected
        expectedStr <- paste(expectedStr, collapse=", ")
        expectedStr <- "(" %++% expectedStr %++% ")"
    }
    else {
        expectedStr <- ifelse(is.null(expected), "NULL", 
                           ifelse(is.character(expected), dQuote(expected), expected))
    }
    
    if (missing(parm)) {
        error <- sprintf("No value specified for parameter %s: Expected %s.", dQuote(parmName), expected)
        bigr.err(logSource, error)
    }

    # Check for NULL equivalency right away
    if (is.null(parm)) {
        if (is.null(expected))
            return (T)
    } else {
        if (parm %in% expected)
            return (T)
    }
    
    specified <- ifelse(is.null(parm), "NULL", 
                       ifelse(is.character(parm), dQuote(parm), parm))
    
    error <- sprintf("Invalid value for parameter %s: Expected %s, specified %s. ",
                     dQuote(parmName), expectedStr, specified)
    bigr.err(logSource, error)
    return (F)
}

#'
#' .bigr.compare
#' 
#' Perform comparisons between two objects. The typical use case for this function
#' involves passing in one R object (say, a data.frame), and another BigR object (e.g.,
#' a bigr.frame). It internally converts the BigR object into its equivalent R form,
#' and invokes the compare() function to do the actual work. compare() comes to us from
#' the "compare" package on CRAN.
#' 
#' This is a utility function intended to be used during unit testing / FVT / etc. 
#' It is not a part of the core BigR functionality.
#' 
#' @param a                 first R or BigR object
#' @param b                 second R or BigR object
#' @param compareOutputs    if T, invoke "show" on both objects and compare the stdout
#' @param ...               arguhemts to be passed to compare()
#' @keywords internal

.bigr.compare <- function(a, b, compareOutputs = T, ...) {
    require(compare)
    
    capture <- function(expr) {
        paste(capture.output(expr), collapse="\n")
    }
    
    cls <- class(a)
    a2 <- a
    if (cls == "bigr.frame")
        a2 <- as.data.frame(a)
    else if (cls == "bigr.vector")
        a2 <- as.vector(a)
    
    cls <- class(b)
    b2 <- b
    if (cls == "bigr.frame")
        b2 <- as.data.frame(b)
    else if (cls == "bigr.vector")
        b2 <- as.vector(b)
    
    # First compare data structures
    cmpdata <- compare(a2, b2, ...)
    xcmpdata <- cmpdata$result
    
    # Grab outputs
    xcmpoutput <- T
    if (compareOutputs) {
        outa <- capture(show(a))
        outb <- capture(show(b))
        #cat("outa: ", outa, "\n")
        #cat("outb: ", outb, "\n")
        
        cmpoutput <- compare(outa, outb)
        xcmpoutput <- cmpoutput$result
    }
    
    if (xcmpdata == F | xcmpoutput == F) {
        warning(sprintf("Difference: %s vs. %s (Data: %s, Output: %s)", 
                        deparse(substitute(a)),
                        deparse(substitute(b)),
                        ifelse(xcmpdata, "similar", "different"),
                        ifelse(xcmpoutput, "similar", "different")),
                call.=F, immediate.=F)
        warning(paste("Data comparison flags: ", capture(print(cmpdata))), call.=F)
        if (compareOutputs) {
            warning(paste("Output comparison flags: ", capture(print(cmpoutput))), call.=F)
        }
        
        cat("---", deparse(substitute(a)), "\n")
        print(a)
        cat("---", deparse(substitute(b)), "\n")
        print(b)
    }
}
