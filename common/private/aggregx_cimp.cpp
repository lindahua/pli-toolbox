/**********************************************************
 *
 *  aggregx_cimp.cpp
 *
 *  C++ mex implementation of aggregx
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>
#include "aggreg_base.h"

using namespace lmat;
using namespace lmat::matlab;

// aggregation functors

struct aggreg_by_scalars { };
struct aggreg_by_rows { };
struct aggreg_by_columns { };


template<typename T, class AOp>
inline marray do_aggreg(aggreg_by_scalars, 
        const index_t K, 
        const cref_matrix<T>& vals, 
        const cref_matrix<int32_t>& I, 
        AOp op)
{
    
    marray mR = marray::numeric_matrix<T>(K, 1);
    ref_col<T> R = view_as_col<T>(mR);
    
    op.init(R.nelems(), R.ptr_data());
    
    const index_t N = vals.nelems();
    
    for (index_t i = 0; i < N; ++i)
    {
        int32_t k = I[i];
        if (k >= 0 && k < K)
        {
            op(R[k], vals[i]);
        }
    }
    
    return mR;
}


template<typename T, class AOp>
inline marray do_aggreg(aggreg_by_rows, 
        const index_t K, 
        const cref_matrix<T>& vals, 
        const cref_matrix<int32_t>& I, 
        AOp op)
{            
    const index_t m = vals.nrows();
    const index_t n = vals.ncolumns();
    
    marray mR = marray::numeric_matrix<T>(K, n);
    ref_matrix<T> R = view2d<T>(mR);    
    
    op.init(R.nelems(), R.ptr_data());

    for (index_t j = 0; j < n; ++j)
    {
        const T* v = vals.ptr_col(j);
        T *r = R.ptr_col(j);

        for (index_t i = 0; i < m; ++i)
        {
            int32_t k = I[i];
            if (k >= 0 && k < K)
            {
                op(r[k], v[i]);
            }
        }
    }
    
    return mR;
}


template<typename T, class AOp>
inline marray do_aggreg(aggreg_by_columns, 
        const index_t K, 
        const cref_matrix<T>& vals, 
        const cref_matrix<int32_t>& I, 
        AOp op)
{
    const index_t m = vals.nrows();
    const index_t n = vals.ncolumns();
    
    marray mR = marray::numeric_matrix<T>(m, K);
    ref_matrix<T> R = view2d<T>(mR);
    
    op.init(R.nelems(), R.ptr_data());
    
    for (index_t j = 0; j < n; ++j)
    {
        index_t k = I[j];
        
        if (k >= 0 && k < K)
        {
            const T* v = vals.ptr_col(j);
            T* r = R.ptr_col(k);
            
            for (index_t i = 0; i < m; ++i)
            {
                op(r[i], v[i]);
            }
        }
    }
    
    return mR;
}


template<typename T, typename Ag>
inline marray do_aggreg(int op_code, Ag ag,
        const index_t K, 
        const cref_matrix<T>& vals, 
        const cref_matrix<int32_t>& I)
{
    marray ret(0);
    
    switch (op_code)
    {
        case 0:
            ret = do_aggreg(ag, K, vals, I, aop_sum<T>());
            break;
        case 1:
            ret = do_aggreg(ag, K, vals, I, aop_max<T>());
            break;
        case 2:
            ret = do_aggreg(ag, K, vals, I, aop_min<T>());  
            break;
    }
    
    return ret;
}



struct Program
{
    const index_t K;
    const_marray mVals;
    const_marray mI;
    const int op_code;
    
    marray mA;
    
    Program(const mxArray *prhs[])
    : K(const_marray(prhs[0]).scalar<int32_t>())
    , mVals(prhs[1])
    , mI(prhs[2])
    , op_code((int)(const_marray(prhs[3]).get_scalar()))
    , mA(0)
    {
    }
    
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
        
        // dispatch
        
        const index_t m = vals.nrows();
        const index_t n = vals.ncolumns();
        
        const index_t Im = I.nrows();
        const index_t In = I.ncolumns();
        
        if (Im == m && In == n)  // aggregate scalars
        {
            mA = do_aggreg(op_code, aggreg_by_scalars(), 
                    K, vals, I);                                    
        }
        else if (Im == m && In == 1) // aggregate rows
        {
            mA = do_aggreg(op_code, aggreg_by_rows(), 
                    K, vals, I);             
        }
        else if (Im == 1 && In == n) // aggregate columns
        {
            mA = do_aggreg(op_code, aggreg_by_columns(), 
                    K, vals, I);             
        }
        else
        {
            mexErrMsgIdAndTxt("aggregx:invalidarg", 
                    "The size of I is invalid.");
        }
    }        
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    dispatch_for_fptypes(prg);
    
    plhs[0] = prg.mA;
}





