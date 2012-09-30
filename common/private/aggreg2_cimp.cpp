/**********************************************************
 *
 *  aggreg2_cimp.cpp
 *
 *  C++ mex implementation of aggreg2
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>
#include "aggreg_base.h"

using namespace lmat;
using namespace lmat::matlab;


inline char get_form_sym(const index_t m, const index_t n, 
        const cref_matrix<int32_t>& I)
{
    char c = char(0);
    
    const index_t Im = I.nrows();
    const index_t In = I.ncolumns();
    
    if (Im == m && In == n)
    {
        c = 'm';
    }
    else if (Im == m && In == 1)
    {
        c = 'c';
    }
    else if (Im == 1 && In == n)
    {
        c = 'r';
    }
    else
    {
        mexErrMsgIdAndTxt("aggreg2:invalidarg", 
                "The size of I or J is invalid.");
    }
    
    return c;
}


template<typename T, class AOp>
LMAT_ENSURE_INLINE        
inline void aggreg_to(AOp op, 
        const index_t K, const index_t L, ref_matrix<T>& A, 
        const index_t k, const index_t l, const T& x)
{
    if (k >= 0 && k < K && l >= 0 && l < L)
    {
        op(A(k, l), x);
    }
}


template<typename T, class AOp>
void aggreg2(
        const cref_matrix<T>& vals,
        const cref_matrix<int32_t>& I, 
        const cref_matrix<int32_t>& J, 
        ref_matrix<T>& A, AOp op)
{
    const index_t m = vals.nrows();
    const index_t n = vals.ncolumns();
    const index_t K = A.nrows();
    const index_t L = A.ncolumns();
        
    char sI = get_form_sym(m, n, I);
    char sJ = get_form_sym(m, n, J);
    
    op.init(A.nelems(), A.ptr_data());
    
    if (sI == 'm')
    {
        if (sJ == 'm')
        {
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I(i, j), J(i, j), vals(i, j));
                }
            }
        }
        else if (sJ == 'c')
        {
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I(i, j), J[i], vals(i, j));
                }
            }            
        }
        else
        {                        
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I(i, j), J[j], vals(i, j));
                }
            }
        }
    }
    else if (sI == 'c')
    {
        if (sJ == 'm')
        {
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I[i], J(i, j), vals(i, j));
                }
            }
        }
        else if (sJ == 'c')
        {
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I[i], J[i], vals(i, j));
                }
            }            
        }
        else
        {                        
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I[i], J[j], vals(i, j));
                }
            }
        }        
    }
    else
    {
        if (sJ == 'm')
        {
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I[j], J(i, j), vals(i, j));
                }
            }
        }
        else if (sJ == 'c')
        {
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I[j], J[i], vals(i, j));
                }
            }            
        }
        else
        {                        
            for (index_t j = 0; j < n; ++j)
            {
                for (index_t i = 0; i < m; ++i)
                {
                    aggreg_to(op, K, L, A, 
                            I[j], J[j], vals(i, j));
                }
            }
        }         
    }
}


struct Program
{
    const index_t am;
    const index_t an;
    const_marray mVals;
    const_marray mI;
    const_marray mJ;
    const int op_code;
    
    marray mA;
    
    Program(const mxArray *prhs[])
    : am(const_marray(prhs[0]).scalar<int32_t>())
    , an(const_marray(prhs[1]).scalar<int32_t>())
    , mVals(prhs[2])
    , mI(prhs[3])
    , mJ(prhs[4])
    , op_code((int)(const_marray(prhs[5]).get_scalar()))
    , mA(0)
    { }
    
    mxClassID class_id() const
    {
        return mVals.class_id();
    }
    
    template<typename T>
    void run() 
    {
        // prepare inputs
        
        cref_matrix<T> vals = view2d<T>(mVals);
        cref_matrix<int32_t> I = view2d<int32_t>(mI);
        cref_matrix<int32_t> J = view2d<int32_t>(mJ);
        
        // prepare outputs
        
        mA = marray::numeric_matrix<T>(am, an);
        ref_matrix<T> A = view2d<T>(mA);
        
        // main
        
        switch (op_code)
        {
            case 0:
                aggreg2(vals, I, J, A, aop_sum<T>());
                break;
            case 1:
                aggreg2(vals, I, J, A, aop_max<T>());
                break;
            case 2:
                aggreg2(vals, I, J, A, aop_min<T>());
                break;
        }
    }
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    dispatch_for_fptypes(prg);
    
    plhs[0] = prg.mA;
}


