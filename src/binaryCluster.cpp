// [[Rcpp::depends(RcppArmadillo)]]
// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

# include <RcppArmadillo.h>
# include <memory>
using namespace Rcpp ;


///////////////////////////////////////
// Function to randomly assign covariates to the various views, for initialisation
// (e.g. randomly assigns some covariates as "on" and others as "off")
///////////////////////////////////////
arma::umat randomlyAssignCovariatesToViews(const int& nCovariates, const int& nViews)
{
  arma::umat covariatesOnInEachView(nViews, nCovariates);
  arma::uvec initialIndices = arma::regspace<arma::uvec>(0, nCovariates - 1);
  initialIndices = arma::shuffle(initialIndices);
  
  unsigned int startInd = 0;
  unsigned int endInd1  = floor(double(nCovariates)/double(nViews));
  unsigned int endInd   = endInd1 - 1;
  
  covariatesOnInEachView.zeros();
  
  for(int k = 0; k<nViews; k++){
    
    if(k == (nViews - 1))
    {
      endInd = nCovariates - 1;
    }
    
    for(int i = startInd; i <= endInd; i++)
    {
      covariatesOnInEachView(k,initialIndices(i)) = 1;
    }
    
    if(k != (nViews - 1))
    {
      startInd = endInd + 1;
      endInd   += endInd1;
    }
    
  }
  
  return covariatesOnInEachView;
}

///////////////////////////////////////
// Function to calculate log gamma function of all elements in a vector
///////////////////////////////////////
arma::rowvec rowveclgammaf(const arma::rowvec& alpha){
  arma::rowvec outvec(alpha.n_elem);
  for(int k = 0; k<alpha.n_elem; k++){
    outvec(k) = lgamma(alpha(k));
  }
  return outvec;
}

///////////////////////////////////////
// Function to calculate log gamma function of all elements in a matrix
///////////////////////////////////////
arma::mat matlgammaf(const arma::mat& A){
  arma::mat outMat(A.n_rows, A.n_cols);
  for(int k = 0; k<A.n_rows; k++){
    outMat.row(k) = rowveclgammaf(A.row(k));
  }
  return outMat;
}


///////////////////////////////////////
// data class -- just a useful way of storing the data & associated info 
//               (e.g. number of covariates, number of categories, ...)
///////////////////////////////////////

class data {
public:
  unsigned int nLevels;   // the number of categories
  int nCovariates, nIndividuals; // number of columns and rows of data matrix
  arma::umat dataSet;     // the data matrix
  arma::uvec dataLevels;  // the data categoriues
  
  // Constructor to set dimensions & defaults:
  data(const arma::umat& X, const unsigned int &nDataLevels)
    : dataSet(X.n_rows, X.n_cols)// matrix member initialization
    , dataLevels(nDataLevels)
  {
    nCovariates  = dataSet.n_cols;
    nIndividuals = dataSet.n_rows;
    dataSet      = X;
    // Categorical data.  We assume the same number of categories for each covariate
    dataLevels = arma::unique(dataSet);
    nLevels    = nDataLevels;
  }
};

///////////////////////////////////////
// priorForCategoricalDatasets class
///////////////////////////////////////

class priorForCategoricalDatasets {
public:
  unsigned int nLevels;
  int dd;
  double aa;
  
  arma::mat beta;
  double sumBeta;
  arma::rowvec gammalnsumBetaMinussumgammalnbeta;
  
  // Constructor to set dimensions & defaults:
  priorForCategoricalDatasets(const data& dataSet)
    : beta(dataSet.nLevels,  dataSet.nCovariates)
    , gammalnsumBetaMinussumgammalnbeta(dataSet.nCovariates)
  {
    dd = dataSet.nCovariates;
    nLevels = dataSet.nLevels;
    // Empirical prior, based on proportions of each category in the (unclustered) dataset
    arma::uvec flattenedData = arma::vectorise( dataSet.dataSet );
    arma::uvec dataCounts = arma::histc(flattenedData, dataSet.dataLevels);
    arma::vec dataProportions = arma::conv_to<arma::vec>::from(dataCounts)/arma::accu(dataCounts);
    arma::vec betaVec = 0.5*nLevels*dataProportions;
    arma::rowvec sumgammalnbeta(dd);
    sumBeta = accu(betaVec);
    sumgammalnbeta.fill(arma::sum(rowveclgammaf(betaVec.t())));
    beta = arma::repmat(betaVec, 1, dd);
    gammalnsumBetaMinussumgammalnbeta = lgammaf(sumBeta) - sumgammalnbeta;
  }
};


///////////////////////////////////////
// component class -- really just a virtual class, from which the more specific component
//                    classes inherit
///////////////////////////////////////

class component {
public:
  int dd, nn;
  double Z0; 
  
  // Constructor to set dimensions & defaults:
  component(){}
  
  virtual void print()
  {
    std::cout << "Default component class\n";
    
  }
  
  // Method to add an item/individual/observation to a component:
  virtual void addItem(const data& dataSet, const int& index, const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior, const unsigned int& calculateZ0flag){}
  
  // Method to remove an item/individual/observation from a component:
  virtual void deleteItem(const data& dataSet, const int& index, const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior, const unsigned int& calculateZ0flag){}
  
  // Method to calculate the log marginal likelihood for the current component: 
  virtual double ZZ(const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior){return(0.0);}
  
  // Method to calculate the change in log marginal likelihood that would result from adding individual with index "index" to the current component 
  virtual double logPredictive(const data& dataSet, const int& index, const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior){return(0.0);}
  
  // Method to return the log marginal likelihood
  virtual arma::rowvec returnLogMarginalLikelihoodVector(){return(arma::zeros<arma::vec>( dd));}
  
};


///////////////////////////////////////
// Categorical class -- this is the Categorical datatype component class
///////////////////////////////////////
class Categorical: public component {
public:
  arma::umat dataCounts;
  arma::rowvec Z0vec;
  arma::mat lgammafMatrix;
  double lgammafnnPlusSumBeta;
  
  // Constructor to set dimensions & defaults:
  Categorical(const priorForCategoricalDatasets& myPrior, const arma::uvec& selectedCovariates)
    : component()
    , dataCounts(myPrior.nLevels, myPrior.dd)
    , lgammafMatrix(myPrior.nLevels, myPrior.dd)
  {
    dd = myPrior.dd;
    dataCounts.zeros();
    lgammafMatrix = matlgammaf(myPrior.beta); 
    lgammafnnPlusSumBeta = lgammaf(myPrior.sumBeta);
    nn = 0;
    Z0 = ZZ(selectedCovariates, myPrior);
  }
  
  void print()
  {
    std::cout << "Categorical component class." << "\n";
    
  }
  
  // Method to calculate the log marginal likelihood for the current component: 
  double ZZ(const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior)
  {
    Z0vec = sum(lgammafMatrix);
    Z0vec = Z0vec + myPrior.gammalnsumBetaMinussumgammalnbeta;
    Z0vec = Z0vec - lgammafnnPlusSumBeta;
    
    return(arma::accu(Z0vec(find(selectedCovariates == 1))));
//    return(arma::accu(Z0vec));
  }
  
  // Method to add an item/indiviudal/observation to a component:
  void addItem(const data& dataSet, const int& index, const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior, const unsigned int& calculateZ0flag)
  {
    arma::urowvec xx = dataSet.dataSet.row(index);
    arma::uword currentCount;//, currentInd;
    lgammafnnPlusSumBeta += log(nn+myPrior.sumBeta); 
    nn++;

    for(int i = 0; i < dd; i++)
    {
      currentCount = dataCounts(xx(i), i);
      lgammafMatrix(xx(i), i) += log(currentCount+ myPrior.beta(xx(i), i));
    }
    
    for(int i = 0; i < dd; i++)
    {
      currentCount = dataCounts(xx(i), i);
      dataCounts(xx(i), i) = currentCount + 1;
    }
    
    if(calculateZ0flag == 1)
    {
      Z0 = ZZ(selectedCovariates, myPrior);
    }
  }
  
  void deleteItem(const data& dataSet, const int& index, const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior, const unsigned int& calculateZ0flag)
  {
    arma::urowvec xx = dataSet.dataSet.row(index);
    arma::uword currentCount;
    nn--;
    lgammafnnPlusSumBeta -= log(nn+myPrior.sumBeta); 

    for(int i = 0; i < dd; i++)
    {
      currentCount = dataCounts(xx(i), i) -1;
      lgammafMatrix(xx(i), i) -= log(currentCount+ myPrior.beta(xx(i), i));
      dataCounts(xx(i), i) = currentCount;
    }


    if(calculateZ0flag == 1)
    {
      Z0 = ZZ(selectedCovariates, myPrior);
    }
  }
  
  
  double logPredictive(const data& dataSet, const int& index, const arma::uvec& selectedCovariates, const priorForCategoricalDatasets& myPrior)
  {
    double ll, llBefore, llAfter;
    int nnsaved = nn;
    arma::rowvec Z0vecsaved     = Z0vec;
    arma::umat dataCountssaved  = dataCounts;
    arma::mat lgammafMatrixsaved = lgammafMatrix;
    double lgammafnnPlusSumBetaSaved = lgammafnnPlusSumBeta;

    // Log likelihood before adding the new item:
    llBefore = Z0;
    
    // Now add the new item (this is a dummy add -- we will remove again shortly)    
    addItem(dataSet, index, selectedCovariates, myPrior, 1);
    //Log likelihood after adding the new item:
    llAfter = Z0;
    
    // Undo the add (making use of saved quantities)
    nn = nnsaved;
    Z0vec = Z0vecsaved;
    dataCounts = dataCountssaved;
    lgammafMatrix = lgammafMatrixsaved;
    lgammafnnPlusSumBeta = lgammafnnPlusSumBetaSaved;
    Z0 = llBefore;
    ll =   llAfter - llBefore;
    return(ll);
  }
  
  arma::rowvec returnLogMarginalLikelihoodVector()
  {
    return(Z0vec);  
  }
  
  
};


class dpm {
public:
  unsigned int viewNumber;
  int KK, NN, dd;
  double aa, a0, b0; 
  arma::uvec zz, nn;
  std::vector<std::shared_ptr<component>> vectorOfComponents; 
  std::string pathToFile;
  arma::uvec selectedCovariates; 
  
  // Constructor to set dimensions & defaults:
  dpm(data dataSet, const priorForCategoricalDatasets& myPrior, unsigned int initialNumberOfComponents, arma::uvec initialClusterAllocations, arma::uvec initialCovariateSelections, unsigned int uniqueIdentfier, std::string const &fileLocation)
    : zz(dataSet.nIndividuals)
    , nn(initialNumberOfComponents)
    , selectedCovariates(dataSet.nCovariates)
  {
    
    viewNumber = uniqueIdentfier;
    selectedCovariates = initialCovariateSelections;
    KK = initialNumberOfComponents;
    NN = dataSet.nIndividuals;
    aa = 1;
    a0 = 2; b0 = 4;
    nn.zeros();
    zz = initialClusterAllocations;
    dd = dataSet.nCovariates;
    
    for(int k = 0; k < (KK + 1); k++)
    {
      addEmptyComponent(myPrior);
    }
    
    
    
    for(int k = 0; k < NN; k++)
    {
      vectorOfComponents[zz(k)]->addItem(dataSet, k, selectedCovariates, myPrior, 0);
      nn[zz(k)] += 1;
    }
    
    forceLogMarginalLikelihoodRecalculation(myPrior);
    
  }
  
  arma::uvec returnConponentAllocations()
  {
    return(zz);
  }
  
  double returnAlpha()
  {
    return(aa);
  }
    
  void addEmptyComponent(const priorForCategoricalDatasets& myPrior)
  {
    auto myCategorical = std::make_shared<Categorical>(myPrior, selectedCovariates);
    vectorOfComponents.push_back(myCategorical);
  }
  
  arma::rowvec gatherMarginalLikelihoods()
  {
    arma::rowvec accumulatedMarginalLikelihood(dd);
    arma::rowvec currentMarginalLikelihood(dd);
    accumulatedMarginalLikelihood.zeros();
    for(int k = 0; k < KK; k++)
    {
      currentMarginalLikelihood     = vectorOfComponents[k]->returnLogMarginalLikelihoodVector();
      accumulatedMarginalLikelihood = accumulatedMarginalLikelihood + currentMarginalLikelihood;
    }
    return(accumulatedMarginalLikelihood);
  }
  
  void forceLogMarginalLikelihoodRecalculation(const priorForCategoricalDatasets& myPrior)
  {
    for(int k = 0; k < (KK + 1); k++)
    {
      vectorOfComponents[k]->ZZ(selectedCovariates, myPrior);
    }
  }
  
  void setSelectedCovariates(  arma::uvec covariateSelections)
  {
    selectedCovariates = covariateSelections;
  }
  
  void sampleAlpha()
  {
    //Sample a new alpha according to the method of Escobar and West, 1995
    
    double u, v1, v2, alpha, A, B, a, b, eta, pi_eta;
    int n, k;
    // For ease of reference, use variable names that are the same as in the paper
    alpha = aa;
    a = a0;
    b = b0;
    n = NN;
    k = KK;

    v1 = arma::randg<arma::vec>( 1, arma::distr_param((alpha+1.0),1.0) )[0];
    v2 = arma::randg<arma::vec>( 1, arma::distr_param(n,1) )[0];
    eta = v1/(v1+v2);
    
    //Escobar & West gives (pi_eta/(1 - pi_eta)) = A/B, where A and B are:
    A    = a + k - 1;
    B    = n*(b - log(eta));
    
    //Rearranging, we have pi_eta = A/(A+B)
    pi_eta     = A/(A + B);
    
    u = arma::randu();
    if (u < pi_eta)
    {
      alpha = arma::randg<arma::vec>(1, arma::distr_param(a+k  , 1/(b - log(eta)) )  )[0];
    }
    else
    {
      alpha = arma::randg<arma::vec>(1, arma::distr_param(a+k-1, 1/(b - log(eta)) )  )[0];
    }
    aa = alpha;
  }

  void gibbs(const unsigned int& numIter, const data& dataSet, const priorForCategoricalDatasets& myPrior)
  {
    arma::vec savedAlphas(numIter+1);
    arma::umat savedClusterLabels(NN, numIter+1);

    savedAlphas(0) = aa;
    savedClusterLabels.col(0) = zz;

    for(int iter = 0; iter < numIter; iter++)
    {
      // Update the component allocations:
      sampleAllocations(dataSet, myPrior);

      //Sample the DP precision parameter:
      sampleAlpha();
      
      savedAlphas(iter+1) = aa;
      savedClusterLabels.col(iter+1) = zz;
      
    }
    if(numIter > 1)
    {
      std::string fileName1, fileName2;
      fileName1 = pathToFile+"savedAlphas" + std::to_string(viewNumber)  + ".csv";
      fileName2 = pathToFile+"savedClusterLabels" + std::to_string(viewNumber) + ".csv";
      savedAlphas.save(fileName1, arma::csv_ascii);
      savedClusterLabels.save(fileName2, arma::csv_ascii);
    }
  }
  
  void sampleAllocations(data dataSet, const priorForCategoricalDatasets& myPrior)
  {
    int kk;
    arma::uvec ind;
    arma::uvec uitemToAppend(1);
    arma::vec  itemToAppend(1);
    arma::vec cumsumVector;
    arma::vec allocationProbabilities(KK+1);
    arma::vec logPrior(KK+1); 
    uitemToAppend.zeros();
    itemToAppend.zeros();
    logPrior(arma::span(0,KK-1))  =  arma::log(arma::conv_to<arma::vec>::from(nn));
    logPrior(KK) = log(aa);
    
    double uu;
    
    
    for(int ii =0; ii < NN; ii++ )
      {
      kk      = zz(ii); //current component label
      // Remove ii-th data item from current component
      nn(kk) -= 1;
      vectorOfComponents[kk]->deleteItem(dataSet, ii, selectedCovariates, myPrior, 1);
      
      if(nn(kk) == 0)
      {
        KK = KK - 1;
        vectorOfComponents.erase(vectorOfComponents.begin()+kk);
        nn.shed_row(kk);
        logPrior.shed_row(kk);
        allocationProbabilities.shed_row(kk);
        ind = find(zz > kk); 
        zz(ind) = zz(ind) - 1; 
      }
      else
      {
        logPrior(kk) = log(nn(kk));
      }
      
      for(int jj = 0; jj < (KK + 1); jj ++)
      {
        allocationProbabilities(jj) = logPrior(jj) + vectorOfComponents[jj]->logPredictive(dataSet, ii, selectedCovariates, myPrior); 
      }
      allocationProbabilities = arma::exp(allocationProbabilities - arma::max(allocationProbabilities));
      allocationProbabilities = allocationProbabilities/arma::accu(allocationProbabilities);
      uu = arma::randu();
      cumsumVector = arma::cumsum(allocationProbabilities);
      kk = arma::accu(uu>cumsumVector);
      
      if(kk == KK)
      {
        KK++;
        nn.insert_rows(kk, uitemToAppend);
        logPrior.insert_rows(kk, itemToAppend);
        allocationProbabilities.insert_rows(kk, itemToAppend);
        addEmptyComponent(myPrior);
      }
      
      zz(ii) = kk;
      nn(kk) += 1;
      logPrior(kk) = log(nn(kk));
      vectorOfComponents[kk]->addItem(dataSet, ii, selectedCovariates, myPrior, 1);
    }
  }
};
  
  
class multiview {
  public:
    unsigned int sampleViewsFlag, hasNullViewFlag;
    int nIndividuals, nCovariates, nViews;
    double aa, a0, b0;
    std::string pathToFile;
    std::vector<std::shared_ptr<dpm>> vectorOfViews; 
    arma::umat covariatesOnInEachView;
    arma::uvec nn, zz;
    
    // Constructor to set dimensions & defaults:
    multiview(const data& dataSet, const priorForCategoricalDatasets& myPrior, const unsigned int& numberOfViews, const unsigned int& initialNumberOfComponents, const unsigned int& hasNull, std::string const &fileLocation)
      :covariatesOnInEachView(numberOfViews, dataSet.nCovariates)
      ,nn(numberOfViews)
      ,zz(dataSet.nCovariates)
    {
      nIndividuals = dataSet.nIndividuals;
      nCovariates  = dataSet.nCovariates;
      nViews = numberOfViews;
      pathToFile = fileLocation;
      arma::uvec initialClusterAllocations(nIndividuals);
      aa = 0.1;
      a0 = 2;
      b0 = 4;
      hasNullViewFlag = hasNull;
      
      // Randomly assign covariates to views  
      if(nViews > 1)
      {
        covariatesOnInEachView = randomlyAssignCovariatesToViews(nCovariates, nViews);
      }
      else
      {
        covariatesOnInEachView.ones();
      }

      arma::uvec zeroToNviewsMinus1Vector = arma::regspace<arma::uvec>(0, nViews - 1);
      for(int k = 0; k < nCovariates; k++)
      {
        zz(k) = arma::accu(zeroToNviewsMinus1Vector % covariatesOnInEachView.col(k));
      }
      
      
      arma::uvec initialCovariateSelections(nCovariates);
      
      for(int k = 0; k < nViews; k++)
      {
        initialCovariateSelections = covariatesOnInEachView.row(k).t();
        nn(k) = arma::accu(initialCovariateSelections);
        
        if(k==0 && hasNullViewFlag == 1)
        {
          initialClusterAllocations.zeros();
        }
        else
        {
          initialClusterAllocations = arma::randi<arma::uvec>( nIndividuals, arma::distr_param(0,initialNumberOfComponents-1) ) ;
        }
        addEmptyView(dataSet, myPrior, initialNumberOfComponents, initialClusterAllocations, initialCovariateSelections, k); 
      }
    }  
    
    void addEmptyView(const data& dataSet, const priorForCategoricalDatasets& myPrior, const int& initialNumberOfComponents, const arma::uvec& initialClusterAllocations, const arma::uvec& initialCovariateSelections, const unsigned int& viewNumber )
    {
      auto myDpm = std::make_shared<dpm>(dataSet, myPrior, initialNumberOfComponents, initialClusterAllocations, initialCovariateSelections, viewNumber, pathToFile);
      vectorOfViews.push_back(myDpm);
      
    }
    
    void gibbs(const unsigned int& numIter, const data& dataSet, const priorForCategoricalDatasets& myPrior)
    {
      arma::uvec zzView(nIndividuals);
      double aaView;
      arma::mat savedAlphaViews(numIter, nViews);
      arma::vec savedAlphas(numIter+1);
      arma::vec savedAlpha_eachView(numIter);
      arma::umat savedClusterLabels(nCovariates, numIter+1);
      arma::ucube savedClusterLabelsViews(nIndividuals, numIter, nViews);
      savedAlphas(0) = aa;
      savedClusterLabels.col(0) = zz;
      
      int numClusteringIterations = 1;
      
      for(int i = 0; i < numIter; i++)
      {
        if ( (i+1) % 100 == 0 )
        {
          std::cout << "Iteration number: " << (i + 1) << "\n" ;
        }
        //First sample the component allocations in each view:
        for(int k = 0; k < nViews; k++)
        {
          if(k==0 && hasNullViewFlag == 1)
          {
            // We do not need to update the null view
          }
          else
          {
            for(int j=0; j < numClusteringIterations; j++)
            {
              vectorOfViews[k]->gibbs(1, dataSet, myPrior);
            }

          }
          aaView = vectorOfViews[k]->returnAlpha();
          savedAlphaViews(i,k) = aaView;
          zzView = vectorOfViews[k]->returnConponentAllocations();
          savedClusterLabelsViews.slice(k).col(i) = zzView;
        }
        
        if(nViews > 1)
        {
        //Now sample the view allocations:
        sampleViewAllocations(myPrior);
         
        //Now sample the Dirichlet precision parameter:
        sampleAlpha();
        }
        savedAlphas(i+1) = aa;
        savedClusterLabels.col(i+1) = zz;
      }
      
      if(nViews > 1)
      {
        savedAlphas.save(pathToFile+"savedViewAlphas.csv", arma::csv_ascii);
        savedClusterLabels.save(pathToFile+"savedViewLabels.csv", arma::csv_ascii);
      }
      
      std::string fileName1, fileName2;
      for(int k = 0; k < nViews; k++)
      {
        fileName1 = pathToFile+"savedAlphas" + std::to_string(k)  + ".csv";
        fileName2 = pathToFile+"savedClusterLabels" + std::to_string(k) + ".csv";
        savedAlpha_eachView = savedAlphaViews.col(k);
        savedAlpha_eachView.save(fileName1, arma::csv_ascii);
        savedClusterLabelsViews.slice(k).save(fileName2, arma::csv_ascii);
      }
      
      
}
    
    void sampleAlpha()
    {
      //Sample a new alpha according to the method of Escobar and West, 1995
      
      double u, v1, v2, alpha, A, B, a, b, eta, pi_eta;
      int n, k;
      arma::uvec uniquezz = unique(zz);
      
      // For ease of reference, use variable names that are the same as in the paper
      alpha = aa;
      a = a0;
      b = b0;
      n = nCovariates;
      k = uniquezz.n_elem; 
      
      v1 = arma::randg<arma::vec>( 1, arma::distr_param((alpha+1.0),1.0) )[0];
      v2 = arma::randg<arma::vec>( 1, arma::distr_param(n,1) )[0];
      eta = v1/(v1+v2);
      
      //Escobar & West gives (pi_eta/(1 - pi_eta)) = A/B, where A and B are:
      A    = a + k - 1;
      B    = n*(b - log(eta));
      
      //Rearranging, we have pi_eta = A/(A+B)
      pi_eta     = A/(A + B);
      
      u = arma::randu();
      if (u < pi_eta)
      {
        alpha = arma::randg<arma::vec>(1, arma::distr_param(a+k  , 1/(b - log(eta)) )  )[0];
      }
      else
      {
        alpha = arma::randg<arma::vec>(1, arma::distr_param(a+k-1, 1/(b - log(eta)) )  )[0];
      }
      
      aa = alpha;
    }
    
    
    void sampleViewAllocations(const priorForCategoricalDatasets& myPrior)
    {
      int kk;
      double uu;
      arma::mat logMarginalLikelihoodMatrix(nViews, nCovariates);
      arma::rowvec currentLogMarginalLikelihoodsForView(nCovariates);
      arma::vec currentLogMarginalLikelihoodsForCovariate(nViews);
      arma::vec logPrior(nViews);
      arma::vec allocationProbabilities(nViews);
      arma::vec cumsumVector(nViews);
      arma::uvec selectedCovariates(nCovariates); 
      
      //pp = log(aa/KK + nn);
      
      for(int ii = 0; ii < nViews; ii++)
      {
        currentLogMarginalLikelihoodsForView = vectorOfViews[ii]->gatherMarginalLikelihoods();
        logMarginalLikelihoodMatrix.row(ii)  = currentLogMarginalLikelihoodsForView;
      }
      
      
      for(int ii = 0; ii < nCovariates ; ii++)
      {
        //Remove current covariate from its view
        kk       = zz(ii);
        nn(kk)   = nn(kk) - 1;
        covariatesOnInEachView(kk, ii) = 0;
        
        logPrior = arma::log(arma::conv_to<arma::vec>::from(nn) + (aa/nViews));
        currentLogMarginalLikelihoodsForCovariate = logMarginalLikelihoodMatrix.col(ii);
        allocationProbabilities = logPrior + currentLogMarginalLikelihoodsForCovariate;
        allocationProbabilities = arma::exp(allocationProbabilities - arma::max(allocationProbabilities));
        allocationProbabilities = allocationProbabilities/arma::accu(allocationProbabilities);
        
        uu           = arma::randu();
        cumsumVector = arma::cumsum(allocationProbabilities);
        kk           = arma::accu(uu>cumsumVector);
         
        zz(ii)   = kk;
        nn(kk)   = nn(kk) + 1;
        covariatesOnInEachView(kk, ii) = 1;
      }
      //Update covariate selection indictator vectors in each view, and recalculate logMarginalLikelihoods
      for(int ii = 0; ii < nViews; ii++)
      {
        selectedCovariates = covariatesOnInEachView.row(ii).t(); 
        vectorOfViews[ii]->setSelectedCovariates(selectedCovariates);
        vectorOfViews[ii]->forceLogMarginalLikelihoodRecalculation(myPrior);
      }
    }
      
    
    
    
};

// [[Rcpp::export()]]
List binaryCluster (const arma::umat X, const int initialNumberOfComponents, const int numIterations, unsigned int nDataLevels, unsigned int nViews, unsigned int hasNull, std::string const &fileLocation) {

  // Initialise the data object:
  data     myData(X, nDataLevels);
  
  // Initialise the prior:
  priorForCategoricalDatasets myPrior(myData);

  // Initialise the multiview object:
  multiview myMultiview(myData, myPrior, nViews, initialNumberOfComponents, hasNull, fileLocation);
  // ... and run for numIterations Gibbs sampling iterations:
  myMultiview.gibbs(numIterations, myData, myPrior);
    
  List ret ;
  ret["successfulCompletion"]     = 1;

  return(ret) ;
}
