/**********************************************************
 *
 *  intcount2_cimp.cpp
 *
 *  C++ mex implementation of intcount (two-subscript case)
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;


struct Program
{
    int32_t m;
    int32_t n;
    const_marray mI;
    const_marray mJ;
    
    marray mC;
   
    Program(const mxArray *prhs[])
    : m(const_marray(prhs[0]).scalar<int32_t>())
    , n(const_marray(prhs[1]).scalar<int32_t>())
    , mI(prhs[2])
    , mJ(prhs[3])
    , mC(0)
    {}
    
    mxClassID class_id() const
    {
        return mI.class_id();
    }
    
    template<typename T>
    void run()
    {
        // prepare input
        
        cref_col<T> I = view_as_col<T>(mI);             
        cref_col<T> J = view_as_col<T>(mJ);
        
        index_t len = I.nelems();
        
        
        // prepare output
        
        mC = marray::numeric_matrix<int32_t>(m, n);
        ref_matrix<int32_t> C = view2d<int32_t>(mC);
        
        // main work
        
        for (index_t idx = 0; idx < len; ++idx)
        {
            int32_t i = I[idx] - 1;
            int32_t j = J[idx] - 1;
            
            if (i >= 0 && i < m && j >= 0 && j < n)
            {
                ++ C(i, j);
            }
        }
    }
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    dispatch_for_numtypes(prg);
    
    plhs[0] = prg.mC;
}

