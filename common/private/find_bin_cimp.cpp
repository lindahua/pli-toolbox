/**********************************************************
 *
 *  intcount1_cimp.cpp
 *
 *  C++ mex implementation of find_bin 
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;


template<typename T>
inline void find_bin(int32_t K, const T *edges, 
        index_t n, const T *x, int32_t *r)
{
    for (index_t i = 0; i < n; ++i)
    {
        int32_t k = 0;
        
        const T xi = x[i];        
        while (k < K && xi >= edges[k]) ++k; 
        r[i] = k;
    }
}


template<typename T>
inline void find_bin_sorted(int32_t K, const T *edges,
        index_t n, const T *x, int32_t *r)
{
    int32_t k = 0;
 
    for (index_t i = 0; i < n; ++i)
    {
        const T xi = x[i];        
        while (k < K && xi >= edges[k]) ++k;
        r[i] = k;        
    }        
}


struct Program
{
    const_marray mEdges;
    const_marray mX;
    const bool is_sorted;
    
    marray mR;
    
    Program(const mxArray *prhs[])
    : mEdges(prhs[0])
    , mX(prhs[1])
    , is_sorted(const_marray(prhs[2]).scalar<bool>())
    , mR(0)
    { }
    
    mxClassID class_id() const
    {
        return mEdges.class_id();
    }
    
    template<typename T>
    void run()
    {
        int32_t K = (int32_t)mEdges.nelems();
        const T *edges = mEdges.data<T>();
        
        index_t n = mX.nelems();
        const T *x = mX.data<T>();
        
        mR = marray::numeric_matrix<int32_t>(mX.nrows(), mX.ncolumns());
        int32_t *r = mR.data<int32_t>();
        
        if (is_sorted)
        {
            find_bin_sorted(K, edges, n, x, r);
        }
        else
        {
            find_bin(K, edges, n, x, r);
        }
    }            
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);    
    
    dispatch_for_numtypes(prg);    
    
    plhs[0] = prg.mR;
}






