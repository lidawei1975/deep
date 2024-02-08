def remove_digital_filter(dic, data, truncate=True, post_proc=False):
    """
    Remove the digital filter from Bruker data.

    Parameters
    ----------
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray
        Array of NMR data to remove digital filter from.
    truncate : bool, optional
        True to truncate the phase shift prior to removing the digital filter.
        This typically produces a better looking spectrum but may remove
        useful data.  False uses a non-truncated phase.
    post_proc : bool, optional
        True if the digitial filter is to be removed post processing, i.e after
        fourier transformation. The corrected FID will not be returned, only a
        corrected spectrum in the frequency dimension will be returned

    Returns
    -------
    ndata : ndarray
        Array of NMR data with digital filter removed

    See Also
    ---------
    rm_dig_filter : Remove digital filter by specifying parameters.

    """
    if 'acqus' not in dic:
        raise ValueError("dictionary does not contain acqus parameters")

    if 'DECIM' not in dic['acqus']:
        raise ValueError("dictionary does not contain DECIM parameter")
    decim = dic['acqus']['DECIM']

    if 'DSPFVS' not in dic['acqus']:
        raise ValueError("dictionary does not contain DSPFVS parameter")
    dspfvs = dic['acqus']['DSPFVS']

    if 'GRPDLY' not in dic['acqus']:
        grpdly = 0
    else:
        grpdly = dic['acqus']['GRPDLY']

    return rm_dig_filter(data, decim, dspfvs, grpdly, truncate, post_proc)


def rm_dig_filter(
        data, decim, dspfvs, grpdly=0, truncate_grpdly=True, post_proc=False):
    """
    Remove the digital filter from Bruker data.

    Parameters
    ----------
    data : ndarray
        Array of NMR data to remove digital filter from.
    decim : int
        Decimation rate (Bruker DECIM parameter).
    dspfvs : int
        Firmware version (Bruker DSPFVS parameter).
    grpdly : float, optional
        Group delay. (Bruker GRPDLY parameter). When non-zero decim and
        dspfvs are ignored.
    truncate_grpdly : bool, optional
        True to truncate the value of grpdly provided or determined from
        the decim and dspfvs parameters before removing the digital filter.
        This typically produces a better looking spectrum but may remove useful
        data.  False uses a non-truncated grpdly value.
    post_proc : bool, optional
        True if the digitial filter is to be removed post processing, i.e after
        fourier transformation. The corrected time domain data will not be
        returned, only the corrected spectrum in the frequency dimension will
        be returned

    Returns
    -------
    ndata : ndarray
        Array of NMR data with digital filter removed.

    See Also
    --------
    remove_digital_filter : Remove digital filter using Bruker dictionary.

    """
    # Case I: post_proc flag is set to False (default)
    # This algorithm gives results similar but not exactly the same
    # as NMRPipe.  It was worked out by examining sample FID converted using
    # NMRPipe against spectra shifted with nmrglue's processing functions.
    # When a frequency shifting with a fft first (fft->first order phase->ifft)
    # the middle of the fid nearly matches NMRPipe's and the difference at the
    # beginning is simply the end of the spectra reversed.  A few points at
    # the end of the spectra are skipped entirely.
    # -jjh 2010.12.01

    # The algorithm is as follows:
    # 1. FFT the data
    # 2. Apply a negative first order phase to the data.  The phase is
    #    determined by the GRPDLY parameter or found in the DSPFVS/DECIM
    #    loopup table.
    # 3. Inverse FFT
    # (these first three steps are a frequency shift with a FFT first, fsh2)
    # 4. Round the applied first order phase up by two integers. For example
    #    71.4 -> 73, 67.8 -> 69, and 48 -> 50, this is the number of points
    #    removed from the end of the fid.
    # 5. If the size of the removed portion is greater than 6, remove the first
    #    6 points, reverse the remaining points, and add then to the beginning
    #    of the spectra.  If less that 6 points were removed, leave the FID
    #    alone.
    # -----------------------------------------------------------------------

    # Case II : post_proc flag is True
    # 1. In this case, it is assumed that the data is already fourier
    #    transformed
    # 2. A first order phase correction equal to 2*PI*GRPDLY is applied to the
    #    data and the time-corrected FT data is returned

    # The frequency dimension will have the same number of points as the
    # original time domain data, but the time domain data will remain
    # uncorrected
    # -----------------------------------------------------------------------

    if grpdly > 0:  # use group delay value if provided (not 0 or -1)
        phase = grpdly

    # determine the phase correction
    else:
        if dspfvs >= 14:    # DSPFVS greater than 14 give no phase correction.
            phase = 0.
        else:   # loop up the phase in the table
            if dspfvs not in bruker_dsp_table:
                raise ValueError("dspfvs not in lookup table")
            if decim not in bruker_dsp_table[dspfvs]:
                raise ValueError("decim not in lookup table")
            phase = bruker_dsp_table[dspfvs][decim]

    if truncate_grpdly:     # truncate the phase
        phase = np.floor(phase)

    # and the number of points to remove (skip) and add to the beginning
    skip = int(np.floor(phase + 2.))    # round up two integers
    add = int(max(skip - 6, 0))           # 6 less, or 0

    # DEBUG
    # print("phase: %f, skip: %i add: %i"%(phase,skip,add))

    if post_proc:
        s = data.shape[-1]
        pdata = data * np.exp(2.j * np.pi * phase * np.arange(s) / s)
        pdata = pdata.astype(data.dtype)
        return pdata

    else:
        # frequency shift
        pdata = proc_base.fsh2(data, phase)

        # add points at the end of the specta to beginning
        pdata[..., :add] = pdata[..., :add] + pdata[..., :-(add + 1):-1]
        # remove points at end of spectra
        return pdata[..., :-skip]