/**********************************************************
 *
 *  lbfgs_calcdir_cimp.cpp
 *
 *  C++ mex implementation of the direction computation
 *  part for L-BFGS algorithm
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

typedef cref_matrix<double, 0, 0> cmat_t;
typedef cref_matrix<double, 0, 1> cvec_t;
typedef ref_col<double> vec_t;


/*
 
Reference MATLAB implementation

% backward-pass
q = g;
for j = [mm : -1 : 1, hm : -1 : mm+1]
    talpha(j) = rho(j) * (S(:,j)' * q);
    q = q - talpha(j) * Y(:,j);
end                    

% forward-pass
z = q;
for j = [mm + 1 : hm, 1 : mm]
    beta = rho(j) * (Y(:,j)' * z);
    z = z + (talpha(j) - beta) * S(:,j);
end 

*/


struct LBFGS_Update
{
    cmat_t S;
    cmat_t Y;
    cvec_t rho;
    
    marray mZ;
    vec_t z;
            
    const index_t hm;
    const index_t mm;
    
    LMAT_ENSURE_INLINE
    LBFGS_Update(const mxArray *prhs[])
    : S(view2d<double>(prhs[0]))
    , Y(view2d<double>(prhs[1]))
    , rho(view_as_col<double>(prhs[2]))
    , mZ(duplicate(prhs[3]))
    , z(view_as_col<double>(mZ))
    , hm((index_t)const_marray(prhs[4]).get_scalar())
    , mm((index_t)const_marray(prhs[5]).get_scalar())
    { }
            
    
    LMAT_ENSURE_INLINE
    void backward_step(index_t j, vec_t& q, double *alpha)
    {
        cvec_t s = S.column(j);
        cvec_t y = Y.column(j);
        
        alpha[j] = rho[j] * dot(s, q);
        q -= alpha[j] * y;
    }
    
    
    LMAT_ENSURE_INLINE
    void forward_step(index_t j, vec_t& z, const double *alpha)
    {
        cvec_t s = S.column(j);
        cvec_t y = Y.column(j);
        
        double beta = rho[j] * dot(y, z);
        z += (alpha[j] - beta) * s;
    }
    
    
    void run()
    {
        const index_t d = S.nrows();
        
        dblock<double> alpha_blk(hm);
        double *alpha = alpha_blk.ptr_data();
        
        // backward-pass
        
        vec_t q = z;
        
        for (index_t j = mm - 1; j >= 0; --j)
        {
            backward_step(j, q, alpha);
        }
        
        for (index_t j = hm - 1; j >= mm; --j)
        {
            backward_step(j, q, alpha);
        }
        
        
        // forward-pass
        
        for (index_t j = mm; j < hm; ++j)
        {
            forward_step(j, z, alpha);
        }
        
        for (index_t j = 0; j < mm; ++j)
        {
            forward_step(j, z, alpha);
        }
    }
};




/**
 * Input:
 * [0] S:       history of x differences
 * [1] Y:       history of g differences
 * [2] rho:     vector of 1 ./ (s' * y)
 * [3] g:       new gradient vector, i.e. initial z
 * [4] hm:      history length
 * [5] mm:      middle division
 *
 * Output:
 * [0] z:       evaluated direction
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    LBFGS_Update prg(prhs);
    prg.run();
    plhs[0] = prg.mZ;
}

