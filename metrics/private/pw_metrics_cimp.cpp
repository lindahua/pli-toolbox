/**********************************************************
 *
 *  pw_metrics_cimp.cpp
 *
 *  C++ mex implementation of a set of common metrics:
 *
 *  - pw_cityblock
 *  - pw_chebyshev
 *  - pw_hamming
 *
 **********************************************************/


#include "dist_base.h"
#include <cmath>

using namespace lmat;
using namespace lmat::matlab;


enum DistType
{
    DIST_CITYBLOCK = 1,
    DIST_CHEBYSHEV = 2,
    DIST_HAMMING = 3
};


template<typename T>
struct cityblock
{
    LMAT_ENSURE_INLINE
    T operator() (index_t n, const T *x, const T *y)
    {
        T s(0);
        for (index_t i = 0; i < n; ++i)
            s += std::abs(x[i] - y[i]);
        
        return s;
    }
};


template<typename T>
struct chebyshev
{
    LMAT_ENSURE_INLINE
    T operator() (index_t n, const T *x, const T *y)
    {
        T s(0);
        for (index_t i = 0; i < n; ++i)
        {
            T v = std::abs(x[i] - y[i]);
            if (v > s) s = v;
        }
        
        return s;
    }
};


template<typename T>
struct hamming
{
    LMAT_ENSURE_INLINE
    T operator() (index_t n, const T *x, const T *y)
    {
        T s(0);
        for (index_t i = 0; i < n; ++i)
        {
            if (x[i] != y[i]) ++s;
        }
        
        return s;
    }
};       


struct Program
{
    DistType dist_type;
    const_marray mX;
    const_marray mY;
    
    marray mD;
    
    
    Program(const mxArray *prhs[])
    : dist_type((DistType)(const_marray(prhs[0]).scalar<int32_t>()))
    , mX(prhs[1])
    , mY(prhs[2])
    , mD(0)
    {}
    
    mxClassID class_id() const
    {
        return mX.class_id();
    }
    
    template<typename T>
    void run()
    {       
        switch (dist_type)
        {
            case DIST_CITYBLOCK:
                mD = do_pw_dists<T, cityblock<T> >(
                        cityblock<T>(), mX, mY);
                break;
                
            case DIST_CHEBYSHEV:
                mD = do_pw_dists<T, chebyshev<T> >(
                        chebyshev<T>(), mX, mY);                
                break;
                
            case DIST_HAMMING:                                
                mD = do_pw_dists<T, hamming<T> >(
                        hamming<T>(), mX, mY);
                break;
        }
        
    }       
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    dispatch_for_fptypes(prg);
    
    plhs[0] = prg.mD;
}






