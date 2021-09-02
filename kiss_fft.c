/*
Copyright (c) 2003-2010, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "kiss_fft.h"

#include "_kiss_fft_guts.h"
/* The guts header contains all the multiplication and addition macros that are defined for
 fixed or floating point complex numbers.  It also delares the kf_ internal functions.
 */

static void kf_bfly2(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx * Fout2;
    kiss_fft_cpx * tw1 = st->twiddles;
    kiss_fft_cpx t;
    Fout2 = Fout + m;
    do{
        C_FIXDIV(*Fout,2); C_FIXDIV(*Fout2,2);

        C_MUL (t,  *Fout2 , *tw1);
        tw1 += fstride;
        C_SUB( *Fout2 ,  *Fout , t );
        C_ADDTO( *Fout ,  t );
        ++Fout2;
        ++Fout;
    }while (--m);
}

static void kf_bfly4(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        const size_t m
        )
{
    kiss_fft_cpx *tw1,*tw2,*tw3;
    kiss_fft_cpx scratch[6];
    size_t k=m;
    const size_t m2=2*m;
    const size_t m3=3*m;


    tw3 = tw2 = tw1 = st->twiddles;

    do {
        C_FIXDIV(*Fout,4); C_FIXDIV(Fout[m],4); C_FIXDIV(Fout[m2],4); C_FIXDIV(Fout[m3],4);

        C_MUL(scratch[0],Fout[m] , *tw1 );
        C_MUL(scratch[1],Fout[m2] , *tw2 );
        C_MUL(scratch[2],Fout[m3] , *tw3 );

        C_SUB( scratch[5] , *Fout, scratch[1] );
        C_ADDTO(*Fout, scratch[1]);
        C_ADD( scratch[3] , scratch[0] , scratch[2] );
        C_SUB( scratch[4] , scratch[0] , scratch[2] );
        C_SUB( Fout[m2], *Fout, scratch[3] );
        tw1 += fstride;
        tw2 += fstride*2;
        tw3 += fstride*3;
        C_ADDTO( *Fout , scratch[3] );

        if(st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        }else{
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    }while(--k);
}

static void kf_bfly3(
         kiss_fft_cpx * Fout,
         const size_t fstride,
         const kiss_fft_cfg st,
         size_t m
         )
{
     size_t k=m;
     const size_t m2 = 2*m;
     kiss_fft_cpx *tw1,*tw2;
     kiss_fft_cpx scratch[5];
     kiss_fft_cpx epi3;
     epi3 = st->twiddles[fstride*m];

     tw1=tw2=st->twiddles;

     do{
         C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

         C_MUL(scratch[1],Fout[m] , *tw1);
         C_MUL(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
     }while(--k);
}

static void kf_bfly5(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
    int u;
    kiss_fft_cpx scratch[13];
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx *tw;
    kiss_fft_cpx ya,yb;
    ya = twiddles[fstride*m];
    yb = twiddles[fstride*2*m];

    Fout0=Fout;
    Fout1=Fout0+m;
    Fout2=Fout0+2*m;
    Fout3=Fout0+3*m;
    Fout4=Fout0+4*m;

    tw=st->twiddles;
    for ( u=0; u<m; ++u ) {
        C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
        scratch[0] = *Fout0;

        C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
        C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
        C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
        C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

        C_ADD( scratch[7],scratch[1],scratch[4]);
        C_SUB( scratch[10],scratch[1],scratch[4]);
        C_ADD( scratch[8],scratch[2],scratch[3]);
        C_SUB( scratch[9],scratch[2],scratch[3]);

        Fout0->r += scratch[7].r + scratch[8].r;
        Fout0->i += scratch[7].i + scratch[8].i;

        scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
        scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

        scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
        scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

        C_SUB(*Fout1,scratch[5],scratch[6]);
        C_ADD(*Fout4,scratch[5],scratch[6]);

        scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
        scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
        scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
        scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

        C_ADD(*Fout2,scratch[11],scratch[12]);
        C_SUB(*Fout3,scratch[11],scratch[12]);

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
    }
}

/* perform the butterfly for one stage of a mixed radix FFT */
static void kf_bfly_generic(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m,
        int p
        )
{
    int u,k,q1,q;
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_cpx * scratch = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(sizeof(kiss_fft_cpx)*p);

    for ( u=0; u<m; ++u ) {
        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            scratch[q1] = Fout[ k  ];
            C_FIXDIV(scratch[q1],p);
            k += m;
        }

        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            int twidx=0;
            Fout[ k ] = scratch[0];
            for (q=1;q<p;++q ) {
                twidx += fstride * k;
                if (twidx>=Norig) twidx-=Norig;
                C_MUL(t,scratch[q] , twiddles[twidx] );
                C_ADDTO( Fout[ k ] ,t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}

static
void kf_work(
        kiss_fft_cpx * Fout,
        const kiss_fft_cpx * f,
        const size_t fstride,
        int in_stride,
        int * factors,
        const kiss_fft_cfg st
        )
{
    kiss_fft_cpx * Fout_beg=Fout;
    const int p=*factors++; /* the radix  */
    const int m=*factors++; /* stage's fft length/p */
    const kiss_fft_cpx * Fout_end = Fout + p*m;

#ifdef _OPENMP
    // use openmp extensions at the 
    // top-level (not recursive)
    if (fstride==1 && p<=5)
    {
        int k;

        // execute the p different work units in different threads
#       pragma omp parallel for
        for (k=0;k<p;++k) 
            kf_work( Fout +k*m, f+ fstride*in_stride*k,fstride*p,in_stride,factors,st);
        // all threads have joined by this point

        switch (p) {
            case 2: kf_bfly2(Fout,fstride,st,m); break;
            case 3: kf_bfly3(Fout,fstride,st,m); break; 
            case 4: kf_bfly4(Fout,fstride,st,m); break;
            case 5: kf_bfly5(Fout,fstride,st,m); break; 
            default: kf_bfly_generic(Fout,fstride,st,m,p); break;
        }
        return;
    }
#endif

    if (m==1) {
        do{
            *Fout = *f;
            f += fstride*in_stride;
        }while(++Fout != Fout_end );
    }else{
        do{
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m, 
            // each one takes a decimated version of the input
            kf_work( Fout , f, fstride*p, in_stride, factors,st);
            f += fstride*in_stride;
        }while( (Fout += m) != Fout_end );
    }

    Fout=Fout_beg;

    // recombine the p smaller DFTs 
    switch (p) {
        case 2: kf_bfly2(Fout,fstride,st,m); break;
        case 3: kf_bfly3(Fout,fstride,st,m); break; 
        case 4: kf_bfly4(Fout,fstride,st,m); break;
        case 5: kf_bfly5(Fout,fstride,st,m); break; 
        default: kf_bfly_generic(Fout,fstride,st,m,p); break;
    }
}

/*  facbuf is populated by p1,m1,p2,m2, ...
    where 
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static 
void kf_factor(int n,int * facbuf)
{
    int p=4;
    double floor_sqrt;
    floor_sqrt = floor( sqrt((double)n) );

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}

/*
 *
 * User-callable function to allocate all necessary storage space for the fft.
 *
 * The return value is a contiguous block of memory, allocated with malloc.  As such,
 * It can be freed with free(), rather than a kiss_fft-specific function.
 * */
kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem )
{
    kiss_fft_cfg st=NULL;
    size_t memneeded = sizeof(struct kiss_fft_state)
        + sizeof(kiss_fft_cpx)*(nfft-1); /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( kiss_fft_cfg)KISS_FFT_MALLOC( memneeded );
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        int i;
        st->nfft=nfft;
        st->inverse = inverse_fft;

        for (i=0;i<nfft;++i) {
            const double pi=3.141592653589793238462643383279502884197169399375105820974944;
            double phase = -2*pi*i / nfft;
            if (st->inverse)
                phase *= -1;
            kf_cexp(st->twiddles+i, phase );
        }

        kf_factor(nfft,st->factors);
    }
    return st;
}


void kiss_fft_stride(kiss_fft_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int in_stride)
{
    if (fin == fout) {
        //NOTE: this is not really an in-place FFT algorithm.
        //It just performs an out-of-place FFT into a temp buffer
        kiss_fft_cpx * tmpbuf = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC( sizeof(kiss_fft_cpx)*st->nfft);
        kf_work(tmpbuf,fin,1,in_stride, st->factors,st);
        memcpy(fout,tmpbuf,sizeof(kiss_fft_cpx)*st->nfft);
        KISS_FFT_TMP_FREE(tmpbuf);
    }else{
        kf_work( fout, fin, 1,in_stride, st->factors,st );
    }
}

void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    kiss_fft_stride(cfg,fin,fout,1);
}


void kiss_fft_cleanup(void)
{
    // nothing needed any more
}

int kiss_fft_next_fast_size(int n)
{
    while(1) {
        int m=n;
        while ( (m%2) == 0 ) m/=2;
        while ( (m%3) == 0 ) m/=3;
        while ( (m%5) == 0 ) m/=5;
        if (m<=1)
            break; /* n is completely factorable by twos, threes, and fives */
        n++;
    }
    return n;
}



// 2D part

struct kiss_fftnd_state{
    int dimprod; /* dimsum would be mighty tasty right now */
    int ndims; 
    int *dims;
    kiss_fft_cfg *states; /* cfg states for each dimension */
    kiss_fft_cpx * tmpbuf; /*buffer capable of hold the entire input */
};

// 2D part

kiss_fftnd_cfg kiss_fftnd_alloc(const int *dims,int ndims,int inverse_fft,void*mem,size_t*lenmem)
{
    kiss_fftnd_cfg st = NULL;
    int i;
    int dimprod=1;
    size_t memneeded = sizeof(struct kiss_fftnd_state);
    char * ptr;

    for (i=0;i<ndims;++i) {
        size_t sublen=0;
        kiss_fft_alloc (dims[i], inverse_fft, NULL, &sublen);
        memneeded += sublen;   /* st->states[i] */
        dimprod *= dims[i];
    }
    memneeded += sizeof(int) * ndims;/*  st->dims */
    memneeded += sizeof(void*) * ndims;/* st->states  */
    memneeded += sizeof(kiss_fft_cpx) * dimprod; /* st->tmpbuf */

    if (lenmem == NULL) {/* allocate for the caller*/
        st = (kiss_fftnd_cfg) malloc (memneeded);
    } else { /* initialize supplied buffer if big enough */
        if (*lenmem >= memneeded)
            st = (kiss_fftnd_cfg) mem;
        *lenmem = memneeded; /*tell caller how big struct is (or would be) */
    }
    if (!st)
        return NULL; /*malloc failed or buffer too small */

    st->dimprod = dimprod;
    st->ndims = ndims;
    ptr=(char*)(st+1);

    st->states = (kiss_fft_cfg *)ptr;
    ptr += sizeof(void*) * ndims;

    st->dims = (int*)ptr;
    ptr += sizeof(int) * ndims;

    st->tmpbuf = (kiss_fft_cpx*)ptr;
    ptr += sizeof(kiss_fft_cpx) * dimprod;

    for (i=0;i<ndims;++i) {
        size_t len;
        st->dims[i] = dims[i];
        kiss_fft_alloc (st->dims[i], inverse_fft, NULL, &len);
        st->states[i] = kiss_fft_alloc (st->dims[i], inverse_fft, ptr,&len);
        ptr += len;
    }
    /*
Hi there!
If you're looking at this particular code, it probably means you've got a brain-dead bounds checker 
that thinks the above code overwrites the end of the array.
It doesn't.
-- Mark 
P.S.
The below code might give you some warm fuzzies and help convince you.
       */
    if ( ptr - (char*)st != (int)memneeded ) {
        fprintf(stderr,
                "################################################################################\n"
                "Internal error! Memory allocation miscalculation\n"
                "################################################################################\n"
               );
    }
    return st;
}

/*
 This works by tackling one dimension at a time.
 In effect,
 Each stage starts out by reshaping the matrix into a DixSi 2d matrix.
 A Di-sized fft is taken of each column, transposing the matrix as it goes.
Here's a 3-d example:
Take a 2x3x4 matrix, laid out in memory as a contiguous buffer
 [ [ [ a b c d ] [ e f g h ] [ i j k l ] ]
   [ [ m n o p ] [ q r s t ] [ u v w x ] ] ]
Stage 0 ( D=2): treat the buffer as a 2x12 matrix
   [ [a b ... k l]
     [m n ... w x] ]
   FFT each column with size 2.
   Transpose the matrix at the same time using kiss_fft_stride.
   [ [ a+m a-m ]
     [ b+n b-n]
     ...
     [ k+w k-w ]
     [ l+x l-x ] ]
   Note fft([x y]) == [x+y x-y]
Stage 1 ( D=3) treats the buffer (the output of stage D=2) as an 3x8 matrix,
   [ [ a+m a-m b+n b-n c+o c-o d+p d-p ] 
     [ e+q e-q f+r f-r g+s g-s h+t h-t ]
     [ i+u i-u j+v j-v k+w k-w l+x l-x ] ]
   And perform FFTs (size=3) on each of the columns as above, transposing 
   the matrix as it goes.  The output of stage 1 is 
       (Legend: ap = [ a+m e+q i+u ]
                am = [ a-m e-q i-u ] )
   
   [ [ sum(ap) fft(ap)[0] fft(ap)[1] ]
     [ sum(am) fft(am)[0] fft(am)[1] ]
     [ sum(bp) fft(bp)[0] fft(bp)[1] ]
     [ sum(bm) fft(bm)[0] fft(bm)[1] ]
     [ sum(cp) fft(cp)[0] fft(cp)[1] ]
     [ sum(cm) fft(cm)[0] fft(cm)[1] ]
     [ sum(dp) fft(dp)[0] fft(dp)[1] ]
     [ sum(dm) fft(dm)[0] fft(dm)[1] ]  ]
Stage 2 ( D=4) treats this buffer as a 4*6 matrix,
   [ [ sum(ap) fft(ap)[0] fft(ap)[1] sum(am) fft(am)[0] fft(am)[1] ]
     [ sum(bp) fft(bp)[0] fft(bp)[1] sum(bm) fft(bm)[0] fft(bm)[1] ]
     [ sum(cp) fft(cp)[0] fft(cp)[1] sum(cm) fft(cm)[0] fft(cm)[1] ]
     [ sum(dp) fft(dp)[0] fft(dp)[1] sum(dm) fft(dm)[0] fft(dm)[1] ]  ]
   Then FFTs each column, transposing as it goes.
   The resulting matrix is the 3d FFT of the 2x3x4 input matrix.
   Note as a sanity check that the first element of the final 
   stage's output (DC term) is 
   sum( [ sum(ap) sum(bp) sum(cp) sum(dp) ] )
   , i.e. the summation of all 24 input elements. 
*/
void kiss_fftnd(kiss_fftnd_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    int i,k;
    const kiss_fft_cpx * bufin=fin;
    kiss_fft_cpx * bufout;

    /*arrange it so the last bufout == fout*/
    if ( st->ndims & 1 ) {
        bufout = fout;
        if (fin==fout) {
            memcpy( st->tmpbuf, fin, sizeof(kiss_fft_cpx) * st->dimprod );
            bufin = st->tmpbuf;
        }
    }else
        bufout = st->tmpbuf;

    for ( k=0; k < st->ndims; ++k) {
        int curdim = st->dims[k];
        int stride = st->dimprod / curdim;

        for ( i=0 ; i<stride ; ++i ) 
            kiss_fft_stride( st->states[k], bufin+i , bufout+i*curdim, stride );

        /*toggle back and forth between the two buffers*/
        if (bufout == st->tmpbuf){
            bufout = fout;
            bufin = st->tmpbuf;
        }else{
            bufout = st->tmpbuf;
            bufin = fout;
        }
    }
}
