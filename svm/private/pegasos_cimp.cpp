/********************************************************************
 *
 *  C++ mex implementation of Pegasos for SVM
 *
 ********************************************************************/

#include "svm_sgdx_common.h"

/**
 * w = alpha * w + (eta/k) sum_{i in A} y_i x_i
 *
 * Here, A is the set of support vectors, i.e. u * y < 1 
 */
template<class WVec>
void update_weight_vec(WVec& wvec, double alpha, 
        const double *pX, const double *pY, const index_t k, 
        double *us, double eta)
{
    const index_t d = wvec.dim();     
    
    if (k == 1)
    {         
        cvec_t x(pX, d, 1);
        
        // predict
                       
        us[0] = wvec.predict(x);
        
        // update
        
        if (alpha != 1) wvec *= alpha;
        
        double y = *pY;
        if (y * us[0] < 1.0)
            wvec.add_mul(x, y * eta);
    }
    else
    {
        double r = eta / double(k);
        
        // predict
        
        for (index_t i = 0; i < k; ++i)
        {
            cvec_t x(pX + i * d, d, 1);
            us[i] = wvec.predict(x);
        }
        
        // update
        
        if (alpha != 1) wvec *= alpha;
        
        for (index_t i = 0; i < k; ++i)
        {
            double y = pY[i];           
            
            if (y * us[i] < 1.0)
            {
                cvec_t x(pX + i * d, d, 1);                
                wvec.add_mul(x, y * r);
            }
        }
    }
}


template<class WVec>
class Pegasos
{
public:
    Pegasos(WVec& wvec, double lambda, double t0, index_t maxK) 
    : _wvec(wvec), _lambda(lambda), _time(t0), _us_blk(maxK) { }
                
    LMAT_ENSURE_INLINE
    double learn_rate() const
    {
        return 1.0 / (_lambda * _time);
    }
    
    LMAT_ENSURE_INLINE
    void learn(const double *pX, const double *pY, const index_t k)
    {
        ++ _time;
        
        // add support vectors
        
        double *us = _us_blk.ptr_data();
        double eta = learn_rate();
        
        update_weight_vec(_wvec, 1 - eta * _lambda, 
                pX, pY, k, us, eta);
        
        // rescale w
        
        double s = _wvec.sqnorm() * _lambda;
        if (s > 1)
            _wvec *= (1 / std::sqrt(s));        
    }

private:
    WVec& _wvec;
    const double _lambda;
    
    double _time;
    
    dblock<double> _us_blk;
};


// algorithm codes

const index_t SGDX_PEGASOS = 1;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // arguments
    
    const_marray mX(prhs[0]);
    const_marray mY(prhs[1]);
    const_marray mLam(prhs[2]);
    const_marray mAug(prhs[3]);
    const_marray mT0(prhs[4]);
    const_marray mK(prhs[5]);
    
    marray mW = duplicate(prhs[6]);
    marray mW0 = duplicate(prhs[7]);
    
    // take inputs
    
    cmat_t X = view2d<double>(mX);
    cvec_t Y = view_as_col<double>(mY);
    
    double lambda = mLam.get_scalar();
    double aug = mAug.get_scalar();
    
    double t0 = mT0.get_scalar();
    index_t K = (index_t)mK.get_scalar();
    
    // main
    
    if (aug == 0)
    {
        WeightVec w(mW);
        Pegasos<WeightVec> trainer(w, lambda, t0, K);
        run_sgdx(trainer, X, Y, K);
    }
    else
    {
        WeightVecX w(mW, mW0, aug);
        Pegasos<WeightVecX> trainer(w, lambda, t0, K);
        run_sgdx(trainer, X, Y, K);
    }
    
    // return
    
    plhs[0] = mW;
    plhs[1] = mW0;
}


