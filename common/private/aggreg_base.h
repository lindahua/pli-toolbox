// Basic stuff to support aggregation operation

#ifdef _MSC_VER
#pragma once
#endif

#ifndef PLI_AGGREG_BASE_H
#define PLI_AGGREG_BASE_H

#include <light_mat/common/memory.h>
#include <limits>

// operations

template<typename T>
struct aop_sum
{
    LMAT_ENSURE_INLINE
    void init(const lmat::index_t n, T *x) { }
    
    LMAT_ENSURE_INLINE
    void operator() (T& a, const T& x) { a += x; }
};


template<typename T>
struct aop_max
{
    LMAT_ENSURE_INLINE
    void init(const lmat::index_t n, T *x) 
    {
        T v0 = -std::numeric_limits<T>::infinity();
        lmat::fill_vec(n, x, v0);
    }
    
    LMAT_ENSURE_INLINE
    void operator() (T& a, const T& x) 
    { 
        if (x > a) a = x;
    }
};


template<typename T>
struct aop_min
{
    LMAT_ENSURE_INLINE
    void init(const lmat::index_t n, T *x) 
    {
        T v0 = std::numeric_limits<T>::infinity();
        lmat::fill_vec(n, x, v0);
    }
    
    LMAT_ENSURE_INLINE
    void operator() (T& a, const T& x) 
    { 
        if (x < a) a = x;
    }
};

#endif
