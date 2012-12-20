// Bases for distance calulation

#ifdef _MSC_VER
#pragma once
#endif

#ifndef PLI_DIST_BASE_H
#define PLI_DIST_BASE_H

#include <light_mat/matlab/matlab_port.h>


// generic metric calculaiton functions

// This function assumes 
// - d(x, y) == d(y, x)
// - d(x, x) == 0

template<typename T, class Dist, class SMat, class DMat>
inline void pw_dists(Dist dist,
        const lmat::IRegularMatrix<SMat, T>& X, 
        lmat::IRegularMatrix<DMat, T>& D)
{
    const lmat::index_t d = X.nrows();
    const lmat::index_t n = X.ncolumns();
    
    for (lmat::index_t j = 1; j < n; ++j)
    {
        const T* xj = X.ptr_col(j);
        
        for (lmat::index_t i = 0; i < j; ++i)
        {
            const T* xi = X.ptr_col(i);
            D(i, j) = dist(d, xi, xj);
        }
    }
    
    for (lmat::index_t j = 0; j < n; ++j)
    {
        for (lmat::index_t i = j + 1; i < n; ++i)
        {
            D(i, j) = D(j, i);
        }
    }
}


template<typename T, class Dist, class LMat, class RMat, class DMat>
inline void pw_dists(Dist dist,
        const lmat::IRegularMatrix<LMat, T>& X, 
        const lmat::IRegularMatrix<RMat, T>& Y,
        lmat::IRegularMatrix<DMat, T>& D)
{
    const lmat::index_t d = X.nrows();
    const lmat::index_t m = X.ncolumns();
    const lmat::index_t n = Y.ncolumns();
    
    for (lmat::index_t j = 0; j < n; ++j)
    {
        const T *yj = Y.ptr_col(j);
        
        for (lmat::index_t i = 0; i < m; ++i)
        {
            const T *xi = X.ptr_col(i);
            D(i, j) = dist(d, xi, yj);            
        }
    }
}


template<typename T, class Dist>
inline lmat::matlab::marray do_pw_dists(
        Dist dst, 
        lmat::matlab::const_marray mX, 
        lmat::matlab::const_marray mY)
{
    using lmat::matlab::marray;
    
    marray mD(0);
    
    if (mY.is_empty())
    {
        lmat::cref_matrix<T> X = lmat::matlab::view2d<T>(mX);
        lmat::index_t n = X.ncolumns();
        
        mD = marray::numeric_matrix<T>(n, n);
        lmat::ref_matrix<T> D = lmat::matlab::view2d<T>(mD);
        
        pw_dists(dst, X, D);
    }
    else
    {
        lmat::cref_matrix<T> X = lmat::matlab::view2d<T>(mX);
        lmat::cref_matrix<T> Y = lmat::matlab::view2d<T>(mY);
        
        lmat::index_t m = X.ncolumns();
        lmat::index_t n = Y.ncolumns();
        
        mD = marray::numeric_matrix<T>(m, n);
        lmat::ref_matrix<T> D = lmat::matlab::view2d<T>(mD);
        
        pw_dists(dst, X, Y, D);
    }
    
    return mD;
}


#endif



