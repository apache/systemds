# bigr.ml
#
# This file contains a variety of functions for big data machine learning.
#
# Author: IBM

#' Run Linear Regression on a given bigr.frame. 
#'
#' @param bf a \code{bigr.frame}
#' @return a \code{data.frame} with the ...
#' @family analytics
#bigr.lm <- function(bf) {
bigr.lm <- function(bf, y_index, intercept=F, maxiter=100, tol=1e-3, lambda=1) {
    logSource <- "bigr.lm"
    
    # Validate parameters
    if (.bigr.isNullOrEmpty(bf)) {
        bigr.err(logSource, "lm can only be invoked on a bigr.frame.")
        return(NULL)
    }
    if (class(bf) != bigr.env$FRAME_CLASS_NAME) {
        bigr.err(logSource, "lm can only be invoked on a bigr.frame.")
        return(NULL)
    }
    
    temp_dir_table = .bigr.executeJaqlQuery(jaqlExpression = "HadoopTemp().location",
                                            colnames = c("ret"),
                                            coltypes = c("string"),
                                            limit=F)
    #print(temp_dir_table)
    
    temp_dir = temp_dir_table[1,1]
    #temp_dir = "hdfs:/user/biadmin/statiko/bigR/temp"
    #print(paste("Temporary Working Directory: ", temp_dir, sep=""))
    
    # Persist the data frame
    file_prefix <- "input.frame"
    # location of data file in JSON format
    json_data_loc = paste(temp_dir, "/", file_prefix, ".json", sep="")
    
    # Recode input big data frame
    #temp_dir <- "hdfs:/user/biadmin/statiko/bigR/temp"
    recode_and_dummycode = ml.recode(bf, y_index, temp_dir, file_prefix)
    ml_data_loc = recode_and_dummycode$ml_data_loc
    new_label_index = recode_and_dummycode$new_label_index
    #ml_data_loc = "hdfs:/user/biadmin/statiko/bigR/temp/input.frame.ml"
    #print(paste("After recode: data_loc = ", ml_data_loc, sep=""))
    
    # Execute lm
    
    results_file = paste(temp_dir,"/weights.mtx",sep="")
    intercept_int = if(intercept) 1 else 0
    lm_strs_w_args <- paste("lm(\'",ml_data_loc,"\',",new_label_index,",",intercept_int,",",maxiter,",",tol,",",lambda,",\'",results_file,"\')",sep="")
    #print(lm_strs_w_args)
    
    .bigr.executeJaqlQuery(jaqlExpression = lm_strs_w_args, 
                           colnames=c("ret"),
                           coltypes=c("numer"), 
                           limit=FALSE)
    
    decode_wts_str <- paste("decodeWeights(\'",results_file,"\',\'",json_data_loc,"\',",y_index,")",sep="")
    #print(decode_wts_str)
    df <- .bigr.executeJaqlQuery(jaqlExpression = decode_wts_str, 
                                 colnames=c("attribute", "value", "parameter"),
                                 coltypes=c("string","string","numer"), 
                                 limit=FALSE)
    
    .bigr.executeJaqlQuery(jaqlExpression = paste("cleanupML(\'",results_file,"\')",sep=""),
                           colnames = c("ret"),
                           coltypes = c("numer"),
                           limit = F)
    
    return(df)
    
    
    #     X_file = "hdfs:/user/biadmin/sen/bigR/data.mtx"
    #     y_file = "hdfs:/user/biadmin/sen/bigR/labels.mtx"
    #     results_file = paste(data_dir,"/weights.mtx",sep="")
    #     print(results_file)
    #     
    #     intercept_int = if(intercept) 1 else 0
    #     
    #     lm_strs_w_args <- paste("lm(\'",X_file,"\',\'",y_file,"\',",intercept_int,",",maxiter,",",tol,",",lambda,",\'",results_file,"\')",sep="")
    #     print(lm_strs_w_args)
    #     
    #     df <- bigr.executeJaqlQuery(jaqlExpression = lm_strs_w_args, 
    #                                     colnames=c("ret"),
    #                                     coltypes=c("numer"), 
    #                                     limit=FALSE)
    #     
    #     bigr.executeJaqlQuery(jaqlExpression = paste("cleanupML(\'",results_file,"\')",sep=""),
    #                           colnames = c("ret"),
    #                           coltypes = c("numer"),
    #                           limit = F)
    #     
    #     return(df)
}

ml.recode = function(bf, y_index, tempdir, file_prefix) {
    
    # location of data file in csv format
    #csv_data_loc <- paste(tempdir, "/", file_prefix, ".csv", sep="")
    # location of data file in JSON format
    json_data_loc = paste(tempdir, "/", file_prefix, ".json", sep="")
    # location of json attribute metadata file
    json_mtd_loc = paste(tempdir, "/", file_prefix, ".meta.json", sep="")
    # location of the data in systemml format
    ml_data_loc  <- paste(tempdir, "/", file_prefix, ".ml", sep="")
    ml_data_mtd_loc<- paste(ml_data_loc, ".mtd", sep="")
    recode_map_dir <- paste(tempdir, "/", file_prefix, ".map/", sep="")
    
    # Persist the big data frame to temporary location for recoding
    bigr.persist(bf, bigr.env$JSON_FILE, json_data_loc)
    
    # ----------------------------------------
    # Generate initial attribute metadata file
    # ----------------------------------------
    #cnames = "[" %+% paste("'" %+% colnames(bf) %+% "'", collapse=", ") %+% "]" #colnames(df)
    cnames = paste("['", paste(colnames(bf), collapse="', '"), "']", sep="")
    #ctypes = "[" %+% paste("'" %+% coltypes(bf) %+% "'", collapse=", ") %+% "]" #coltypes(df)
    ctypes = paste("['", paste(coltypes(bf), collapse="', '"), "']", sep="")
    genmtd_str <- paste("genMTDJson(", cnames, ",", ctypes, ",\'", json_mtd_loc, "\')", sep="")
    
    start = proc.time()
    begin = proc.time()
    
    #print(genmtd_str)
    #print("---------")
    .bigr.executeJaqlQuery( jaqlExpression = genmtd_str, 
                            colnames = c("ret"),
                            coltypes = c("numeric"),
                            limit = FALSE )
    
    t = (proc.time() - begin)
    begin = proc.time()
    print(paste("genMTDJson: ", t, sep=""))
    
    # ----------------------------------------
    # Generate metadata file, and convert csv file into json format
    # ----------------------------------------
    genRecodeMaps_str <- paste("genRecodeMaps(\'", json_data_loc, "\',\'", json_mtd_loc, "\',\'", recode_map_dir, "\')", sep="")
    #print(genRecodeMaps_str)
    .bigr.executeJaqlQuery( jaqlExpression = genRecodeMaps_str, 
                                 colnames = c("ret"),
                                 coltypes = c("numeric"),
                                 limit = FALSE )
    
#     inputFormat="CSV"
#     genmtd_str <- paste("genMetadata(\'", csv_data_loc, "\',\'", json_mtd_loc, "\',\'", json_data_loc, "\')", sep="")
#     #print(genmtd_str)
#     .bigr.executeJaqlQuery( jaqlExpression = genmtd_str, 
#                             colnames = c("ret"),
#                             coltypes = c("numeric"),
#                             limit = FALSE )
#     t = (proc.time() - begin)
#     begin = proc.time()
#     print(paste("genMetadata: ", t, sep=""))
#     
#     #print("---------")
#     # ----------------------------------------
#     # Enhance metadata
#     # ----------------------------------------
#     toRecode = "yes"       # flag to indicate whether or not to recode
#     inputRecodeMapDir = "" # directory that points to any existing recode maps
#     enhmtd_str <- paste("enhanceMTD(\'", json_data_loc, "\',\'", toRecode, "\',\'", inputRecodeMapDir, "\')", sep="")
#     #print(enhmtd_str)
#     .bigr.executeJaqlQuery( jaqlExpression = enhmtd_str, 
#                             colnames = c("ret"),
#                             coltypes = c("numeric"),
#                             limit = FALSE )
    t = (proc.time() - begin)
    begin = proc.time()
    print(paste("genRecodeMaps: ", t, sep=""))
    
    #print("---------")
    # ----------------------------------------
    # Recode 
    # ----------------------------------------
    recodeAttrs = "" # List of attributes to recode, empty string means all relevant attributes
    recode_str <- paste("recodeBigFrame(\'", json_data_loc, "\',\'", recodeAttrs, "\')", sep="")
    #print(recode_str)
    .bigr.executeJaqlQuery( jaqlExpression = recode_str, 
                            colnames = c("ret"),
                            coltypes = c("numeric"),
                            limit = FALSE )
    
    t = (proc.time() - begin)
    begin = proc.time()
    print(paste("recodeBigFrame: ", t, sep=""))
    
    # ----------------------------------------
    # Dummycode
    # ----------------------------------------
    dummycode_str <- paste("dummy_code(\'", json_data_loc, "\',", y_index, ",\'", ml_data_loc, "\')", sep="")
    #print(dummycode_str)
    new_label_index_frame <- .bigr.executeJaqlQuery( jaqlExpression = dummycode_str, 
                                                     colnames = c("ret"),
                                                     coltypes = c("string"),
                                                     limit = FALSE )
    new_label_index <- as.numeric(new_label_index_frame[1,1])
    #print(paste("new_label_index", new_label_index))
    
    t = (proc.time() - begin)
    begin = proc.time()
    print(paste("dummy_code: ", t, sep=""))
    
    # ----------------------------------------
    # Generate Matrix Metadata (.mtd file)
    # ----------------------------------------
    genMtxMTD_str <- paste("genMatrixMTD2(\'", json_data_loc, "\',\'", ml_data_mtd_loc, "\')", sep="")
    #print(genMtxMTD_str)
    .bigr.executeJaqlQuery( jaqlExpression = genMtxMTD_str, 
                            colnames = c("ret"),
                            coltypes = c("numeric"),
                            limit = FALSE )
    
    t = (proc.time() - begin)
    begin = proc.time()
    print(paste("genMatrixMTD2: ", t, sep=""))
    
    t = proc.time()-start
    print(paste("TotalTime: ", t, sep=""))
    
    
    #print("---------")
    #unpivot_str <- paste("unpivot(\'", json_data_loc, "\',\'", ml_data_loc, "\')", sep="")
    #print(unpivot_str)
    #bigr.executeJaqlQuery( jaqlExpression = unpivot_str, 
    #                       colnames = c("ret"),
    #                       coltypes = c("numeric"),
    #                       limit = FALSE )
    #print("---------")
    #genMtxMTD_str <- paste("genMatrixMTD(\'", json_data_loc, "\',\'", ml_data_mtd_loc, "\')", sep="")
    #print(genMtxMTD_str)
    #bigr.executeJaqlQuery( jaqlExpression = genMtxMTD_str, 
    #                       colnames = c("ret"),
    #                       coltypes = c("numeric"),
    #                       limit = FALSE )
    #print("---------")
    return(list("new_label_index"=new_label_index, "ml_data_loc"=ml_data_loc))
    
}
