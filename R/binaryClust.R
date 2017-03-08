binaryClust <- function (X, numberOfIterations = 100, pathToFile = getwd(), featureSelection = FALSE, seed = NULL, initialNumberOfComponents = ceiling(nrow(X)/20))
{
  if(any(is.na(X)))
  {
    print("Input data matrix contains NAs.  Must be strictly binary (contain 0s and 1s only).")
  }
  
  if(!identical(as.vector(X),as.numeric(as.logical(X))) )
  {
    print("Data matrix must be strictly binary (contain 0s and 1s only).")
  }
  
  
  if(featureSelection) {
    nViews  <- 2 
    hasNull <- 1
  } else {
    nViews  <- 1 
    hasNull <- 0
  }
  
  
  nIndividuals <- nrow(X)
  nCovariates  <- ncol(X)
  nDataLevels  <- 2

  if(!is.null(seed))
  {
    set.seed(seed)
  }
  
  pathToFile              <- path.expand(pathToFile) # To deal with paths including "~" for home 
    
  nCharactersInPathToFile <- nchar(pathToFile)
  finalCharacter          <- substr(pathToFile, nCharactersInPathToFile, nCharactersInPathToFile)

  if(!identical(finalCharacter, .Platform$file.sep))
  {
    pathToFile <- paste(pathToFile, .Platform$file.sep, sep = "")
  }
  
  if(!dir.exists(pathToFile))
  {
    dir.create(pathToFile, showWarnings = FALSE)
  }
  
  output <- binaryCluster(X, initialNumberOfComponents, numberOfIterations, nDataLevels, nViews, hasNull, pathToFile)
  print(paste("Output written to", pathToFile))
  return(output)
  
}