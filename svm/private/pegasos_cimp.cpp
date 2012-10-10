/************************************************
 *
 *  C++ mex implmentation of Pegasos for Linear SVM
 *
 ************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

struct Program
{        
    // data
    
    const_marray mX;        // the samples
    const_marray mY;        // the responses (labels)
    const_marray mInds;     // the stream of sample index to process   
    
    double lambda;      // the regularization coefficient
    double aug;         // the augmented multiplier
    int32_t base_t;    // the number of iterations done before
    
    // outputs
    
    marray mW;      // the weights
    marray mB;      // the bias term    
       
    
    Program(const mxArray *prhs[])
    : mX(prhs[0])
    , mY(prhs[1])
    , mInds(prhs[2])
    , mW(duplicate(prhs[3]))
    , mB(duplicate(prhs[4]))
    , lambda(const_marray(prhs[5]).get_scalar())
    , aug(const_marray(prhs[6]).get_scalar())
    , base_t(const_marray(prhs[7]).scalar<int32_t>())
    { }
    
    
    // Run
    
    void run()
    {
        // get inputs
        
        cref_matrix<double> X = view2d<double>(mX);
        cref_col<double> Y = view_as_col<double>(mY);
        cref_col<int32_t> I = view_as_col<int32_t>(mInds);
        
        const index_t d = X.nrows();
        const index_t n = X.ncolumns();
        const index_t T = I.nelems();
        
        // prepare outputs
        
        ref_col<double> w = view_as_col<double>(mW);
        double& b = *(mB.data<double>());
        
        // main-loop
        
        for (index_t t = 0; t < T; ++t)
        {
            // learning rate
            double eta = 1.0 / (lambda * double(base_t + t + 1));
            
            // fetch next sample
            index_t idx = I[t];
            
            cref_col<double> x(X.ptr_col(idx), d);
            double y = Y[idx];
            
            // make prediction
            double u = dot(w, x);
            if (aug) u += b * aug;
            
            // update w
            
            double rp = 1 - eta * lambda;
            w *= rp;
            
            if (y * u < 1.0)
            {
                double ey = eta * y;
                w += ey * x;
                
                if (aug) b += ey * aug;
            }
            
            // rescale w
            
            double w_nrm2 = lmat::sqL2norm(w) + b * b;
            double s = w_nrm2 * lambda;
            if (s > 1)
            {
                s = 1 / std::sqrt(s);
                w *= s;
                
                if (aug) b *= s;
            }
        }
    }
            
};



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    prg.run();
    
    plhs[0] = prg.mW;
    plhs[1] = prg.mB;
}



