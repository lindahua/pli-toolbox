/********************************************************************
 *
 *  C++ mex implementation on a Extended SGD for SVM
 *
 ********************************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

// weight vector classes

typedef cref_matrix<double, 0, 1> cvec_t;
typedef cref_matrix<double, 0, 0> cmat_t;

typedef ref_matrix<double, 0, 1> vec_t;
typedef ref_matrix<double, 0, 0> mat_t;


class WeightVec
{
public:
    
    LMAT_ENSURE_INLINE
    WeightVec(marray mW)
    : _w(mW.ptr_real(), mW.nrows(), 1) { }
    
        
    LMAT_ENSURE_INLINE
    index_t dim() const
    {
        return _w.nrows();
    }    
    
    LMAT_ENSURE_INLINE
    double sqnorm() const
    {
        return lmat::sqL2norm(_w);
    }
    
    LMAT_ENSURE_INLINE
    double predict(const cvec_t& x) const
    {
        return lmat::dot(_w, x);
    }    
    
    LMAT_ENSURE_INLINE
    void operator += (const cvec_t& x)
    {
        const index_t d = _w.nrows();
        for (index_t i = 0; i < d; ++i) _w[i] += x[i];
    }
    
    LMAT_ENSURE_INLINE
    void add_mul(const cvec_t& x, double c)
    {
        const index_t d = _w.nrows();
        for (index_t i = 0; i < d; ++i) _w[i] += c * x[i];
    }
    
    LMAT_ENSURE_INLINE
    void operator *= (double c)
    {
        const index_t d = _w.nrows();
        for (index_t i = 0; i < d; ++i) _w[i] *= c;
    }
    
    LMAT_ENSURE_INLINE
    double operator[] (index_t i) const
    {
        return _w[i];
    }
    
    LMAT_ENSURE_INLINE
    double bias() const
    {
        return 0;
    }
        
private:    
    vec_t _w;    
};


class WeightVecX
{
public:
    
    LMAT_ENSURE_INLINE
    WeightVecX(marray mW, marray mW0, double aug)
    : _w(mW.ptr_real(), mW.nrows(), 1) 
    , _w0(*(mW0.ptr_real()))
    , _aug(aug)
    { }
    
    LMAT_ENSURE_INLINE
    index_t dim() const
    {
        return _w.nrows();
    }    
        
    LMAT_ENSURE_INLINE
    double sqnorm() const
    {
        return lmat::sqL2norm(_w) + _w0 * _w0;
    }
    
    LMAT_ENSURE_INLINE
    double predict(const cvec_t& x) const
    {
        return lmat::dot(_w, x) + _w0 * _aug;
    }
    
    
    LMAT_ENSURE_INLINE
    void operator += (const cvec_t& x)
    {
        const index_t d = _w.nrows();
        for (index_t i = 0; i < d; ++i) _w[i] += x[i];
        _w0 += _aug;
    }
    
    LMAT_ENSURE_INLINE
    void add_mul(const cvec_t& x, double c)
    {                        
        const index_t d = _w.nrows();
        for (index_t i = 0; i < d; ++i) _w[i] += c * x[i];
        _w0 += c * _aug;
    }
    
    LMAT_ENSURE_INLINE
    void operator *= (double c)
    {
        const index_t d = _w.nrows();
        for (index_t i = 0; i < d; ++i) _w[i] *= c;
        _w0 *= c;
    }
    
    LMAT_ENSURE_INLINE
    double operator[] (index_t i) const
    {
        return _w[i];
    }
    
    LMAT_ENSURE_INLINE
    double bias() const
    {
        return _w0;
    }
        
private:    
    vec_t _w;  
    double& _w0;
    const double _aug;
};



// main skeleton

template<class Trainer>
void run_sgdx(
        Trainer& trainer,
        const cmat_t& Xstream,
        const cvec_t& Ystream,
        const index_t K)
{
    
    const index_t d = Xstream.nrows();
    const index_t n = Xstream.ncolumns();
        
    index_t rn = n;
    index_t i = 0;
    
    for ( ;rn >= K; rn -= K, i += K)
    {
        const double *pX = Xstream.ptr_col(i);
        const double *pY = Ystream.ptr_data() + i;
        
        trainer.learn(pX, pY, K);
    }
    
    if (rn > 0)
    {                
        const double *pX = Xstream.ptr_col(i);
        const double *pY = Ystream.ptr_data() + i;
        
        trainer.learn(pX, pY, rn);
    }   
    
}


// training algorithms



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
    const_marray mCode(prhs[6]);
    
    marray mW = duplicate(prhs[7]);
    marray mW0 = duplicate(prhs[8]);
    
    // take inputs
    
    cmat_t X = view2d<double>(mX);
    cvec_t Y = view_as_col<double>(mY);
    
    double lambda = mLam.get_scalar();
    double aug = mAug.get_scalar();
    
    double t0 = mT0.get_scalar();
    index_t K = (index_t)mK.get_scalar();
    index_t code = (index_t)mCode.get_scalar();
    
    // main
    
    if (code == SGDX_PEGASOS)
    {
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
    }        
    else
    {
        mexErrMsgIdAndTxt("svm_sgdx_cimp:invalidcode", 
                "The code value is invalid.");
    }
    
    // return
    
    plhs[0] = mW;
    plhs[1] = mW0;
}


