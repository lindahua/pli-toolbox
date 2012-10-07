/**********************************************************
 *
 *  ktop_cimp.cpp
 *
 *  C++ mex implementation of pli_ktop
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

#include <functional>
#include <algorithm>

using namespace lmat;
using namespace lmat::matlab;


template<typename T>
struct idx_lt
{
    const T *x;
    
    LMAT_ENSURE_INLINE
    idx_lt(const T* x_) : x(x_) { }
    
    LMAT_ENSURE_INLINE
    bool operator() (int32_t i, int32_t j) const
    {
        return x[i] < x[j];
    }    
};


template<typename T>
struct idx_gt
{
    const T *x;
    
    LMAT_ENSURE_INLINE
    idx_gt(const T* x_) : x(x_) { }
    
    LMAT_ENSURE_INLINE
    bool operator() (int32_t i, int32_t j) const
    {
        return x[i] > x[j];
    }    
};


template<typename T>
inline void k_smallest(
        index_t n, const T *x, index_t k, int32_t *I, int32_t *ws)
{    
    for (int32_t i = 0; i < n; ++i) ws[i] = i;    
    std::partial_sort(ws, ws+k, ws+n, idx_lt<T>(x));
    
    for (index_t i = 0; i < k; ++i) I[i] = ws[i] + 1;
}


template<typename T>
inline void k_smallest(
        index_t n, const T *x, index_t k, int32_t *I, T *y, int32_t *ws)
{    
    for (int32_t i = 0; i < n; ++i) ws[i] = i;    
    std::partial_sort(ws, ws+k, ws+n, idx_lt<T>(x));
    
    for (index_t i = 0; i < k; ++i) y[i] = x[ws[i]];
    for (index_t i = 0; i < k; ++i) I[i] = ws[i] + 1;
}


template<typename T>
inline void k_largest(
        index_t n, const T *x, index_t k, int32_t *I, int32_t *ws)
{
    for (int32_t i = 0; i < n; ++i) ws[i] = i;    
    std::partial_sort(ws, ws+k, ws+n, idx_gt<T>(x));
   
    for (index_t i = 0; i < k; ++i) I[i] = ws[i] + 1;
}

template<typename T>
inline void k_largest(
        index_t n, const T *x, index_t k, int32_t *I, T *y, int32_t *ws)
{
    for (int32_t i = 0; i < n; ++i) ws[i] = i;    
    std::partial_sort(ws, ws+k, ws+n, idx_gt<T>(x));
   
    for (index_t i = 0; i < k; ++i) y[i] = x[ws[i]];
    for (index_t i = 0; i < k; ++i) I[i] = ws[i] + 1;
}



struct Program
{
    const_marray mX;
    const index_t K;
    const bool is_asc;
    const int dim;
    
    marray mI;
    marray mY;
    const bool calc_Y;
   
    Program(const mxArray *prhs[], bool cy)
    : mX(prhs[0])
    , K(const_marray(prhs[1]).scalar<int32_t>())
    , is_asc(const_marray(prhs[2]).scalar<bool>())
    , dim((int)const_marray(prhs[3]).get_scalar())
    , mI(0)
    , mY(0)
    , calc_Y(cy)
    {}
    
    mxClassID class_id() const
    {
        return mX.class_id();
    }
    
    template<typename T>
    void run()
    {
        cref_matrix<T> X = view2d<T>(mX);
        const index_t m = X.nrows();
        const index_t n = X.ncolumns();
        
        if (dim == 1)
        {
            mI = marray::numeric_matrix<int32_t>(K, n);
            ref_matrix<int32_t> I = view2d<int32_t>(mI);
            
            dblock<int32_t> ws_blk(m);
            int32_t *ws = ws_blk.ptr_data();
            
            if (!calc_Y)
            {
                if (is_asc)
                {                                
                    for (index_t j = 0; j < n; ++j)
                    {
                        k_smallest(m, X.ptr_col(j), K, I.ptr_col(j), ws);
                    }
                }
                else
                {                                
                    for (index_t j = 0; j < n; ++j)
                    {
                        k_largest(m, X.ptr_col(j), K, I.ptr_col(j), ws);
                    }
                }
            }
            else
            {
                mY = marray::numeric_matrix<T>(K, n);
                ref_matrix<T> Y = view2d<T>(mY);
                
                if (is_asc)
                {                                
                    for (index_t j = 0; j < n; ++j)
                    {
                        k_smallest(m, X.ptr_col(j), K, 
                                I.ptr_col(j), Y.ptr_col(j), ws);
                    }
                }
                else
                {                                
                    for (index_t j = 0; j < n; ++j)
                    {
                        k_largest(m, X.ptr_col(j), K, 
                                I.ptr_col(j), Y.ptr_col(j), ws);
                    }
                }
            }
            
        }
        else // dim == 2 && m == 1
        {
            mI = marray::numeric_matrix<int32_t>(1, K);
            
            dblock<int32_t> ws_blk(n);
            int32_t *ws = ws_blk.ptr_data();
            
            if (!calc_Y)
            {
                if (is_asc)
                {
                    k_smallest(n, X.ptr_data(), K, mI.data<int32_t>(), ws);
                }
                else
                {
                    k_largest(n, X.ptr_data(), K, mI.data<int32_t>(), ws);
                }
            }
            else
            {
                mY = marray::numeric_matrix<T>(1, K);                
                
                if (is_asc)
                {
                    k_smallest(n, X.ptr_data(), K, 
                            mI.data<int32_t>(), mY.data<T>(), ws);
                }
                else
                {
                    k_largest(n, X.ptr_data(), K, 
                            mI.data<int32_t>(), mY.data<T>(), ws);
                }                
            }
        }
    }         
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs, nlhs >= 2);
    
    dispatch_for_numtypes(prg);
    
    plhs[0] = prg.mI;
    if (nlhs >= 2)
    {
        plhs[1] = prg.mY;
    }
}




