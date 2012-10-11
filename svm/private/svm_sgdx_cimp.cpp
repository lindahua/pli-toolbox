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
            
    if (K == 1)
    {        
        for (index_t i = 0; i < n; ++i)
        {
            cvec_t x = Xstream.column(i);
            double y = Ystream[i];
            
            trainer.learn(x, y);
        }
    }    
    else
    {    
        index_t rn = n;
        index_t i = 0;
        
        for ( ;rn >= K; rn -= K, i += K)
        {        
            cmat_t X(Xstream.ptr_col(i), d, K);
            cvec_t Y(Ystream.ptr_data() + i, K, 1);
        
            trainer.learn(X, Y);
        }
        
        if (rn > 0)
        {
            range rgn(i, rn);
            
            cmat_t X(Xstream.ptr_col(i), d, rn);
            cvec_t Y(Ystream.ptr_data() + i, rn, 1);
            
            trainer.learn(X, Y);
        }
    }    
    
}


// training algorithms


template<class WVec>
class Pegasos
{
public:
    Pegasos(WVec& wvec, double lambda, index_t t0, index_t maxK) 
    : _wvec(wvec), _lambda(lambda), _time(t0), _us(maxK) { }
    
    void learn(const cvec_t& x, double y)
    {
        ++ _time;
                        
        // make prediction
        
        double u = _wvec.predict(x);
        
        // update w
        
        double eta = learn_rate();
        
        _wvec *= (1 - eta * _lambda); 
        
        if (y * u < 1.0)
            _wvec.add_mul(x, y * eta);
        
        
        // rescale w
        
        double s = _wvec.sqnorm() * _lambda;        
        if (s > 1)
            _wvec *= (1 / std::sqrt(s));        
    }
    
    void learn(const cmat_t& X, const cvec_t& Y)
    {
        ++ _time;
                                
        // make predictions
        
        const index_t k = X.ncolumns();
        for (index_t i = 0; i < k; ++i)
            _us[i] = _wvec.predict(X.column(i));
        
        // update w
        
        double eta = learn_rate(); 
        
        _wvec *= (1 - eta * _lambda);        
        
        double r = eta / double(k);        
        for (index_t i = 0; i < k; ++i)
        {
            double y = Y[i];
            if (y * _us[i] < 1.0) 
                _wvec.add_mul(X.column(i), y * r);
        }
                
        // rescale w
        
        double s = _wvec.sqnorm() * _lambda;        
        if (s > 1)
            _wvec *= (1 / std::sqrt(s)); 
    }
    
    
    LMAT_ENSURE_INLINE
    double learn_rate() const
    {
        return 1.0 / (_lambda * double(_time));
    }
    
    
private:
    WVec& _wvec;
    const double _lambda;
    
    index_t _time;
    
    dblock<double> _us;
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
    
    index_t t0 = (index_t)mT0.get_scalar();
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
