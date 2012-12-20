/**********************************************************
 *
 *  Common devices for implementing SGD for SVM
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>
#include <light_mat/mateval/mat_reduce.h>

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
        return lmat::sqsum(_w);
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
        return lmat::sqsum(_w) + _w0 * _w0;
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

