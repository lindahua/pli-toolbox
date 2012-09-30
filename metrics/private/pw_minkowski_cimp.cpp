/**********************************************************
 *
 *  pw_metrics_cimp.cpp
 *
 *  C++ mex implementation of generic pairwise minkowski
 *
 **********************************************************/


#include "dist_base.h"
#include <cmath>

using namespace lmat;
using namespace lmat::matlab;

template<typename T>
struct minkowski
{
    const T p;
    const T inv_p;
    
    LMAT_ENSURE_INLINE
    minkowski(T p_) : p(p_), inv_p(T(1.0 / p)) { }
    
    LMAT_ENSURE_INLINE
    T operator() (const index_t n, const T *x, const T *y)
    {
        T s(0);
        
        for (index_t i = 0; i < n; ++i)
            s += std::pow(std::abs(x[i] - y[i]), p);
        
        return std::pow(s, inv_p);
    }
};


struct Program
{
    const double p;
    const_marray mX;
    const_marray mY;
    
    marray mD;
    
    Program(const mxArray *prhs[])
    : p(const_marray(prhs[0]).get_scalar())
    , mX(prhs[1])
    , mY(prhs[2])
    , mD(0)
    { }
    
    mxClassID class_id() const
    {
        return mX.class_id();
    }
            
    template<typename T>
    void run()
    {       
        mD = do_pw_dists<T, minkowski<T> >(
                minkowski<T>(T(p)), mX, mY);        
    }
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    dispatch_for_fptypes(prg);
    
    plhs[0] = prg.mD;
}

