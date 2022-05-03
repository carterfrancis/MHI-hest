import nibabel as nb
import numpy as np
import time
import scipy
import scipy.signal
import h5py
from numba import jit, prange

def open_hdf5(hdf5_fname,mode='a'):
    """
    Return a handle to the hdf5 file
    :param hdf5_fname:
    :param mode:
    :return: handle to hdf5 file
    """
    return h5py.File(hdf5_fname, mode=mode)

def close_hdf5(hdf5_handle):
    """
    Flush and close hdf5 file
    :param hdf5_handle:
    :return:
    """
    try:
        fname = hdf5_handle.filename
        hdf5_handle.flush()
        hdf5_handle.close()
        print('File closed: {}'.format(fname))
    except:
        print('Closing file failed. If you provided the correct handle, it was probably already closed.')

def write_mat_hdf5(hdf5_handle, dset, chunksize, dtype=None, dset_name='dset', compression='gzip',
                   compression_opts=6, shuffle=False, fletcher32=True):
    """
    Write dset matrix to hdf5 file in dataset dset_name. Lossless gzip compression.

    :param hdf5_handle:
    :param dset:
    :param dset_name:
    :param compression_opts:
    :param shuffle:
    :param fletcher32:
    :return:
    """

    if dtype is None:
        dtype=dset.dtype
    print('\nAttempting to write to\n\tfile: {}\n\tdset: {}'.format(hdf5_handle.filename.split('/')[-1],dset_name))
    if not dset_name in hdf5_handle:
        dset = hdf5_handle.create_dataset(dset_name, data=dset, dtype=dset.dtype, shape=dset.shape, chunks=chunksize,
                                          compression=compression, compression_opts=compression_opts, shuffle=shuffle, fletcher32=fletcher32)
        print(' Data successfully written.')
        return dset
    else:
        print(' Dataset with name \'{}\' already exists.'.format(dset_name))
        if hdf5_handle[dset_name].shape == dset.shape:
            hdf5_handle[dset_name][...] = dset
            print('  Input data was the same size, so we overwrote it in the hdf5 file.')
        else:
            print('  Input data was not the same size, hdf5 file not overwritten.')

def fit_glm_mat(y,X=None,order=1,axis=-1):
    """
    Detrend timecourse over the given axis with the provided order (0=mean,1=linear,2=quadratic...)
    When X == None, outputs are of decreasing order
    When X != None, column of 1s (mean) is added to the end of the matrix
    :param y:
    :param order:
    :param axis:
    :return:
    """
    nobs = y.shape[axis]
    shape = y.shape

    #just perform the fit with the provided order
    if X is None:
        t = np.arange(nobs) + 1
        X = np.vander(t, order + 1) #results in output of decreasing order (as w/ polyfit etc)
    else:
        t = np.ones(shape[axis]) #for mean
        X = np.vstack((X,t)).T #same here, tack the mean column to the end of the provided matrix
    model, resid, rank, s = np.linalg.lstsq(X, y.T)  # simple linear fit

    return model,resid,rank

#TODO: this could be changed to just mask the data, rather than remove it altogether
def _clean_timecourse(yy,max_data_val=20000,axis=-1):
    """
    remove unreasonable values from 2d data timecourse (2nd dim=time) of multiple voxels
    fills with mean of relevant timeseries
    cleans the data of inf, -inf, nan, and unreasonable large or small vals
    This is tested and works correctly
    """
    inf_bool = ~np.isfinite(yy)
    nan_bool = np.isnan(yy)
    big_bool = yy > max_data_val
    sml_bool = yy < -max_data_val

    if np.sum(inf_bool) > 0 or np.sum(nan_bool) > 0 or np.sum(big_bool) > 0 or np.sum(sml_bool) > 0:
        bad_bool = np.logical_or(inf_bool,np.logical_or(nan_bool,np.logical_or(big_bool,sml_bool)))
        print("    >>>>>>>> Warning: inf or nan detected in your data (filling {} values with timeseries mean) <<<<<<<<<".format(np.sum(bad_bool)))
        yy = np.where(bad_bool, np.ma.array(yy, mask=bad_bool).mean(axis=axis)[..., np.newaxis], yy)
    return yy

def test_window_parameters_for_dfa_blocked(img, mask=None, detrend_type='linear', min_samples=5, max_samples_initial=25,
                                           max_samples_final = 100, sample_steps = 10,
                                           min_window_steps=3, max_window_steps=20, flip_window_when_uneven=False,
                                           by_voxel_group=None, data_type='float32', clean_timecourse_data=False,
                                           verbosity=-1):

    '''
    Calculates the HE and HE_R2 values for a range of sample sizes and numbers of windows. This can help to determine what the optimal fit (HE_R2) is for the datasets that you are using.
    '''
    # pre-calculate the max_samples and window_steps space
    all_samples = np.arange(max_samples_initial, max_samples_final+sample_steps, sample_steps)
    all_windows = np.arange(min_window_steps,max_window_steps+1)

    print("Performing {} iterations of dfa fits".format(all_windows.shape[0]*all_samples.shape[0]))
    # initialise outputs
    all_HE = np.zeros((all_windows.shape[0],all_samples.shape[0]))
    all_R2 = np.zeros((all_windows.shape[0],all_samples.shape[0]))
    all_max_samps = np.zeros(all_samples.shape[0])
    all_wins = np.zeros(all_windows.shape[0])
    all_times = np.zeros((all_windows.shape[0],all_samples.shape[0]))
    all_actual_num_windows = np.zeros((all_windows.shape[0],all_samples.shape[0]))
    all_samples_per_window = []
    all_N_var_mat_per_window = []
    for idx_win, num_window_steps in enumerate(all_windows):
        t_samp_per_window = []
        t_N_var_mat_per_window = []
        for idx_samp, max_samples in enumerate(all_samples):
            st = time.time()
            res = dfa_blocked(img, mask=mask, detrend_type=detrend_type, min_samples=min_samples, max_samples=max_samples, num_window_steps=num_window_steps,
                          manual_sample_nums_list=None, flip_window_when_uneven=flip_window_when_uneven,
                          out_dir=None, by_voxel_group=by_voxel_group,
                          verbosity=-1, output_window_data=True, data_type=data_type,
                          clean_timecourse_data=clean_timecourse_data)
            if isinstance(res['HE'],np.ndarray):
                HE = res['HE']
                HE = np.mean(HE[HE > 0])
                R2 = res['HE_R2']
                R2 = np.mean(R2[R2 > 0])
                N_var_mat = res['var_per_window']
            else:
                HE = res['HE'].get_data()
                HE = np.mean(HE[HE>0])
                R2 = res['HE_R2'].get_data()
                R2 = np.mean(R2[R2 > 0])
                N_var_mat = res['var_per_window'].get_data()
            actual_samples_per_window = res['samples_per_window']
            del res
            print(" Wins/SampMax: {}/{}\tHE: {:.4f} R2: {:.4f}".format(num_window_steps,max_samples,HE, R2))
            print("\t\t\t\t"+str(["{0:0.2f}".format(i) for i in actual_samples_per_window]).replace("'", ""))

            all_HE[idx_win,idx_samp]=HE
            all_R2[idx_win,idx_samp]=R2
            all_max_samps[idx_samp] = max_samples
            all_actual_num_windows[idx_win,idx_samp] = actual_samples_per_window.shape[0]
            all_times[idx_win,idx_samp]=time.time()-st
            t_samp_per_window.append(actual_samples_per_window)
            t_N_var_mat_per_window.append(N_var_mat)
        all_samples_per_window.append(t_samp_per_window)
        all_N_var_mat_per_window.append(t_N_var_mat_per_window)
        all_wins[idx_win] = num_window_steps
    return {'HE':all_HE, 'HE_R2':all_R2, 'max_samples_col':all_max_samps, 'dur_s':all_times,
            'window_steps_row':all_wins,'actual_window_steps':all_actual_num_windows,
            'samples_per_window':all_samples_per_window,'var_per_window':all_N_var_mat_per_window}

def vcorrcoef(X,y):
    """
    Vectorized correlation coefficient calc between single timecourse (y) and multiple timecourses (X)
    Where 1st dim is variable/obs and 2nd dim is time
    from: https://waterprogramming.wordpress.com/2014/06/13/np-vectorized-correlation-coefficient/
    :param X:
    :param y:
    :return:
    """
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def mat_vcorrcoef(X,axis=0):
    """
    correlate each row of X with a flipped version of itself, axis defines the axis over which observations were made
    :param X:
    :return:
    """
    corr = np.zeros((X.shape[axis],X.shape[axis]))
    for vec_idx in range(0,X.shape[axis]):
        corr[vec_idx,:] = vcorrcoef(X, X[vec_idx,:])

    return corr

def mod_corrcoef(X,y):
    all_d = np.zeros((X.shape[0]+1,X.shape[1]))
    all_d[0:X.shape[0],:] = X
    all_d[X.shape[0],:] = y
    return np.corrcoef(all_d)

def ColumnWiseCorrCoef_v1(x,y):
    """
    column-wise correlations (time is in 1st dim)
    :param x:
    :param y:
    :return:
    """
    #https://stackoverflow.com/questions/41538254/memory-efficient-ways-of-computing-large-correlation-matrices
    samples = x.shape[0]
    centered_x = x - np.sum(x, axis=0, keepdims=True) / samples
    centered_y = y - np.sum(y, axis=0, keepdims=True) / samples
    cov_xy = 1./(samples - 1) * np.dot(centered_x.T, centered_y)
    var_x = 1./(samples - 1) * np.sum(centered_x**2, axis=0)
    var_y = 1./(samples - 1) * np.sum(centered_y**2, axis=0)
    return cov_xy / np.sqrt(var_x[:, None] * var_y[None,:])


def matrix_corrcoef_dot(A,B,zscore_data = True):
    """
    Calculate the pair-wise correlation between matrices A and B by calculating the covariance of zscored matrices. This
    method is faster than np.corrcoef as matrix size increases. Data should be zscored along the 2nd dimension (axis=1)
    or zscore_data should be set to True.

    A and B are both time series (rows = regions, cols = timepoints), zscoring performed within function if zscore_data=True
    Returns R matrix of A.shape[0] X B.shape[0]

    :param A: np.ndarray (2d) matrix
            Where rows are regions/voxels, columns are timepoints/observations

    :param B: np.ndarray (2d) matrix
            Same form as A
    :param zscore_data: boolean
            zscore the input data before computing correlation
    :return: correlation matrix of shape A[0]xB[0]
    """

    if zscore_data:
        return np.dot(scipy.stats.zscore(A,axis=1),scipy.stats.zscore(B,axis=1).T)/B.shape[1]
    else:
        return np.dot(A,B.T)/B.shape[1]

# THIS ONLY WORKS FOR A SINGLE VECTOR
def vec_corrcoef_dot(vA,vB,zscore_data=True):
    if zscore_data:
        vA = scipy.stats.zscore(vA)
        vB = scipy.stats.zscore(vB)
    return np.dot(vA, vB) / np.sqrt(
        np.multiply(np.dot(vA, vA), np.dot(vB, vB)))

## this is the current working version of timecourse correlations
def correlate_timecourse_memmaped(img, mask=None, out_dir=None,
                                  global_mean_signal_regression=False, dtype='float16', verbosity=0,
                                  n_vox_per_corr_block = 1000,
                                  absolute_value=True, metric='mean',out_fname_head=None,
                                  CLOBBER=False):
    import tempfile
    import os

    st_orig = time.time()

    img_input_output = True

    # if by_voxel_group is None:  # if none provided, we just make the voxel group arbitrarily large
    by_voxel_group = int(10e10)
    if isinstance(img, str):
        fname = img
        img = nb.load(img)
        print("File image input/output")
    elif isinstance(img,np.ndarray):
        img_input_output = False
        fname = None
    else:
        fname = None

    if out_fname_head is not None:
        fname_head = out_fname_head + "_"
    else:
        fname_head = ""

    try:
        head = img.header  # may not need this in the end
        aff = img.affine
    except:
        if verbosity > -1:
            print("No header / affine in img input, treating as array")
    if mask is None:
        if verbosity > -1:
            print("Don't be lazy, provide a mask please.")
            print('... in the meantime, using non-zero data from first volume to create mask.')
            print('... just because I like you!\n')
        if img_input_output:
            z_mask = np.array(img.dataobj[..., 0]).astype(bool)
        else:
            z_mask = img[...,0][...,None,None].astype(bool)
    else:
        if isinstance(mask, str):
            mask = nb.load(mask)
            z_mask = mask.get_data().astype(bool)
        elif isinstance(mask, np.ndarray):
            if np.ndim(mask) >2:
                z_mask = mask.astype(bool)
            else: #or else we take the first dim, since this should be a list of timecourses
                z_mask = mask.astype(bool)[:,0]

    ## set up the computations for the blocked analyses ##
    # determine where data is in the x,y,z directions via mask
    all_nonzero = np.array(np.where(z_mask))
    min_max = np.vstack((np.min(all_nonzero, axis=1), np.max(all_nonzero, axis=1)))

    # break data into blocks based on the size and the by_voxel_group number
    # fancy indexing does not work, at all, block indexing does, but annoying
    dim_diff = np.diff(min_max, axis=0)[0] + 1
    largest_dim = np.argmax(dim_diff)  # this is the dim that we will split on
    x_idxs = min_max[:, 0]
    y_idxs = min_max[:, 1]
    z_idxs = min_max[:, 2]
    blocks = np.array_split(np.arange(dim_diff[largest_dim]),
                            np.ceil(dim_diff[largest_dim] / float(by_voxel_group)))
    block = blocks[0]

    # set our output arrays that we will fill (weeee)
    corr_out = np.zeros(z_mask.shape).astype(float)
    corr_out_flip = np.zeros(z_mask.shape).astype(float)
    # blk_counter = 0
    print("Computing summary of pair-wise correlations between all time series.")
    print("Iterating over {} block(s) of data".format(len(blocks) ** 3))
    print("Total data shape: {}".format(img.shape))
    print("Total number of voxels within the mask: {}".format(np.sum(z_mask)))

    if global_mean_signal_regression:
        if verbosity > 0:
            print("Global mean signal regression will be performed")
        global_mean_ts = compute_global_mean_signal(img,mask=z_mask)

    print(z_mask.shape)
    # try:
    tf = tempfile.NamedTemporaryFile()
    res = np.memmap(tf, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))
    res = np.ma.masked_array(res,np.eye(res.shape[0],dtype=bool)) #THIS MAY BLOWUP WITH LARGE SIZES

    #this, and the other code above, is legacy code that was used for the HE calculations and worked around here
    x_coords = block + x_idxs[0]
    y_coords = block + y_idxs[0]
    z_coords = block + z_idxs[0]

    if verbosity > 1:
        print("  x,y,z range: {}-{}, {}-{}, {}-{}".format(x_coords[0], x_coords[-1] + 1, y_coords[0],
                                                          y_coords[-1] + 1, z_coords[0], z_coords[-1] + 1))

    # # pull the data from the locations in the mask to determine if we actually have data here
    # if img_input_output:
    #     m_sub = z_mask[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1, z_coords[0]:z_coords[-1] + 1]
    # else:
    #     m_sub = np.squeeze(z_mask[:,0])
    #
    # if np.sum(m_sub) == 0:
    #     if verbosity > 0:
    #         print("  No data here, skipping to next block (this is normal)")
    # else:
    #     pass
    #     # grab between the first and last of the block, extract all values in 4th dim

    if img_input_output:
        d_sub = np.copy(img.get_data()[z_mask,:])
        # d_sub = np.array(img.dataobj[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1,
        #                  z_coords[0]:z_coords[-1] + 1, :])
    else: #we passed a 2d matrix, we just make sure that it doesn't have anything extra going on...
        d_sub = np.copy(np.squeeze(img))
        d_sub = d_sub[z_mask[:,0,0],:] #to only select voxels where there are no zeros in the first frame
        # if clean_timecourse_data:
        #     d_sub = _clean_timecourse(d_sub[m_sub, :])
        # else:
        #     d_sub = d_sub[m_sub, :]

    if global_mean_signal_regression: #this should do global mean signal regression, though not tested against other implementations
        _model, _resid, _rank, _s = np.linalg.lstsq(np.vstack([np.ones_like(global_mean_ts),global_mean_ts]).T, d_sub.T, rcond=None)  # simple linear fit
        _cs, _ms = _model
        _pred = (global_mean_ts[:,None] * _ms + _cs).T
        d_sub = d_sub - _pred

    # subtract means from the input data
    d_sub -= np.mean(d_sub, axis=1)[:, None]

    # normalize the data
    d_sub /= np.sqrt(np.sum(d_sub * d_sub, axis=1))[:, None]

    # from:
    # https://stackoverflow.com/questions/24717513/python-np-corrcoef-memory-error
    # more efficient, potentially :-/
    numrows = d_sub.shape[0]

    if (out_dir is None) and (fname is None):
        print("If you do not have a filename input, you must set an output directory for the h5 matrix file.")
        return 0
    elif out_dir is not None:
        if fname is not None:
            h5_fname = os.path.join(out_dir,fname_head + fname.split('/')[-1].split('.')[0] + "_" + str(numrows) + "els_corr_mat.h5")
        else:
            h5_fname = os.path.join(out_dir, fname_head + str(numrows) + "els_pearson_r_corr_mat.h5")
    elif fname is not None:
        h5_fname = fname.split('.')[0] + str(numrows) + "els_corr_mat.h5"

    if os.path.exists(h5_fname) and not CLOBBER:
        print("Output file already exists, not overwriting because you didn't tell me to\n{}".format(h5_fname))
        return {'h5_fname':h5_fname}
    else:
        print("You have to chosen to write over a previously created output file.")


    #correlate this block, dump to the huge storage array
    r_idx = 0
    if n_vox_per_corr_block > d_sub.shape[0]:
        n_vox_per_corr_block = d_sub.shape[0]
    print('\nIterating processing over {} row chunks'.format(len(range(0, numrows, n_vox_per_corr_block))))
    for r in range(0, numrows, n_vox_per_corr_block):
        r_idx +=1
        st_r = time.time()
        for c in range(0, numrows, n_vox_per_corr_block):
            if c < r:
                pass #we only collect the chunks that cover the upper triangle, since they are the same, we flip the results computed below
            else:
                r1 = r + n_vox_per_corr_block
                c1 = c + n_vox_per_corr_block
                chunk1 = d_sub[r:r1,:]
                chunk2 = (d_sub[c:c1,:]).T
                # print("r: {}\t-\t{}".format(r,r1))
                # print("c: {}\t-\t{}".format(c,c1))
                if c == r:
                    res[r:r1, c:c1] = np.dot(chunk1, chunk2).astype(dtype) # we take them all
                else:
                    res[r:r1, c:c1] = np.dot(chunk1, chunk2).astype(dtype)
                    res[c:c1, r:r1] = np.flipud(np.rot90(res[r:r1, c:c1])) #we compute one and then flip to the other (to make summary easier)

        if verbosity > 1:
            print("{}:{:.2f}s ".format(r_idx,time.time()-st_r),end=' ')
        elif verbosity > -1:
            print(r_idx,end=' ')
    print("")
    print("  Time elapsed for reading data and correlation calculation: {0:.2f}s".format(time.time() - st_orig))

    st = time.time()
    f = open_hdf5(h5_fname)
    if n_vox_per_corr_block < np.sum(z_mask):
        row_chunk = n_vox_per_corr_block
    else:
        row_chunk = np.sum(z_mask)
    print(res.shape)
    print("Attempting to write your data to an hdf5 file")
    #this works if you remove the masking of the array when it was setup, but then the mean calcs don't work
    #dset = write_mat_hdf5(f, res, (row_chunk,np.sum(z_mask)), dset_name="pearson_r", dtype=dtype)
    dset = write_mat_hdf5(f, np.triu(res.astype(dtype)), (row_chunk,np.sum(z_mask)), dset_name="pearson_r", dtype=dtype)
    print("  Time for writing compressed hdf5 file [chunks = ({1},{2})]: {0:.2f}s".format(time.time() - st,row_chunk,np.sum(z_mask)))
    print("  {}".format(h5_fname))

    #compute the summary metric
    if absolute_value:
        print('Using the absolute value of correlations to compute the mean.')
        res = np.abs(res)
    if metric == 'mean':
        #corr_out = (np.sum(res,axis=1)-1)/(res.shape[0]-1) #compute mean by summing, subtracting diagonal, and dividing by number
        # corr_out = np.mean(res,axis=1)
        corr_out = np.zeros(res.shape[0])
        for row_idx in np.arange(0,res.shape[0]):
            corr_out[row_idx] = np.mean(res[row_idx,:])
    corr_out_std = corr_out #temp TODO: update to loop over requested metrics

    del res
    # Turn back into an img
    if img_input_output:
        res_d = np.zeros_like(z_mask).astype(np.float16)
        res_d[z_mask] = corr_out
        head['cal_min'] = np.min(corr_out)
        head['cal_max'] = np.max(corr_out)
        img_corr = nb.Nifti1Image(res_d, aff, header=head)
        try:
            corr_out_name = os.path.join(h5_fname.split('.')[0] + "_mean.nii.gz")
            img_corr.to_filename(corr_out_name)
            print("Image file written to:\n{0}\n".format(corr_out_name))
        except:
                print("Image files were not written to disk properly:\n{0}\n".format(corr_out_name))
    print("Total time elapsed: {0:.2f}s".format(time.time() - st_orig))
    if img_input_output:
        dset2 = write_mat_hdf5(f, img_corr.get_data(), None, dset_name="pearson_r_mean_img",
                              dtype=dtype)
        close_hdf5(f)
        return {'h5_fname':h5_fname,'pearson_r_mean': img_corr}
    else:
        dset2 = write_mat_hdf5(f, corr_out, None, dset_name="pearson_r_mean_vec",
                              dtype=dtype)
        close_hdf5(f)
        return {'h5_fname':h5_fname,'pearson_r_mean': corr_out, 'pearson_r_std':corr_out_std }

    
def correlate_timecourse_memmaped_reduced(img, HE_vec,fit_opts,fit_type,mask=None, out_dir=None,
                                  global_mean_signal_regression=False, dtype='float16', verbosity=0,
                                  n_vox_per_corr_block = 10000,
                                  absolute_value=True, metric='mean',out_fname_head=None,
                                  CLOBBER=False):
    """ working with this version"""
    import tempfile
    import os

    st_orig = time.time()

    img_input_output = True

    # if by_voxel_group is None:  # if none provided, we just make the voxel group arbitrarily large
    by_voxel_group = int(10e10)
    if isinstance(img, str):
        fname = img
        img = nb.load(img)
        print("File image input/output")
    elif isinstance(img,np.ndarray):
        img_input_output = False
        fname = None
    else:
        fname = None

    if out_fname_head is not None:
        fname_head = out_fname_head + "_"
    else:
        fname_head = ""

    try:
        head = img.header  # may not need this in the end
        aff = img.affine
    except:
        if verbosity > -1:
            print("No header / affine in img input, treating as array")
    if mask is None:
        if verbosity > -1:
            print("Don't be lazy, provide a mask please.")
            print('... in the meantime, using non-zero data from first volume to create mask.')
            print('... just because I like you!\n')
        if img_input_output:
            z_mask = np.array(img.dataobj[..., 0]).astype(bool)
        else:
            z_mask = img[...,0][...,None,None].astype(bool)
    else:
        if isinstance(mask, str):
            mask = nb.load(mask)
            z_mask = mask.get_data().astype(bool)
        elif isinstance(mask, np.ndarray):
            if np.ndim(mask) >2:
                z_mask = mask.astype(bool)
            else: #or else we take the first dim, since this should be a list of timecourses
                z_mask = mask.astype(bool)[:,0]

    ## set up the computations for the blocked analyses ##
    # determine where data is in the x,y,z directions via mask
    all_nonzero = np.array(np.where(z_mask))
    min_max = np.vstack((np.min(all_nonzero, axis=1), np.max(all_nonzero, axis=1)))

    # break data into blocks based on the size and the by_voxel_group number
    # fancy indexing does not work, at all, block indexing does, but annoying
    dim_diff = np.diff(min_max, axis=0)[0] + 1
    largest_dim = np.argmax(dim_diff)  # this is the dim that we will split on
    x_idxs = min_max[:, 0]
    y_idxs = min_max[:, 1]
    z_idxs = min_max[:, 2]
    blocks = np.array_split(np.arange(dim_diff[largest_dim]),
                            np.ceil(dim_diff[largest_dim] / float(by_voxel_group)))
    block = blocks[0]

    # set our output arrays that we will fill (weeee)
    corr_out = np.zeros(z_mask.shape).astype(float)
    corr_out_flip = np.zeros(z_mask.shape).astype(float)
    # blk_counter = 0
    print("Computing summary of pair-wise correlations between all time series.")
    print("Iterating over {} block(s) of data".format(len(blocks) ** 3))
    print("Total data shape: {}".format(img.shape))
    print("Total number of voxels within the mask: {}".format(np.sum(z_mask)))

    if global_mean_signal_regression:
        if verbosity > 0:
            print("Global mean signal regression will be performed")
        global_mean_ts = compute_global_mean_signal(img,mask=z_mask)

    print(z_mask.shape)
    # try:
    
    tf = tempfile.NamedTemporaryFile()
    res = np.memmap(tf, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))
    tf3 = tempfile.NamedTemporaryFile()
    pred_r_res = np.memmap(tf3, dtype, mode='w+', shape=(HE_vec.shape[0],HE_vec.shape[0]))

    
    ## take care of generating the meshgrid and then HE values for calculating the predicted correlations
    st_HE = time.time()

    X_HE = np.broadcast_to(HE_vec,(HE_vec.shape[0],HE_vec.shape[0])) #broadcasting is awesome, this is sooo effective
    Y_HE = np.broadcast_to(HE_vec,(HE_vec.shape[0],HE_vec.shape[0])).T
    
    #this, and the other code above, is legacy code that was used for the HE calculations and worked around here
    x_coords = block + x_idxs[0]
    y_coords = block + y_idxs[0]
    z_coords = block + z_idxs[0]

    if verbosity > 1:
        print("  x,y,z range: {}-{}, {}-{}, {}-{}".format(x_coords[0], x_coords[-1] + 1, y_coords[0],
                                                          y_coords[-1] + 1, z_coords[0], z_coords[-1] + 1))
    if img_input_output:
        d_sub = np.copy(img.get_data()[z_mask,:])
        # d_sub = np.array(img.dataobj[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1,
        #                  z_coords[0]:z_coords[-1] + 1, :])
    else: #we passed a 2d matrix, we just make sure that it doesn't have anything extra going on...
        d_sub = np.copy(np.squeeze(img))
        d_sub = d_sub[z_mask[:,0,0],:] #to only select voxels where there are no zeros in the first frame
        # if clean_timecourse_data:
        #     d_sub = _clean_timecourse(d_sub[m_sub, :])
        # else:
        #     d_sub = d_sub[m_sub, :]

    if global_mean_signal_regression: #this should do global mean signal regression, though not tested against other implementations
        _model, _resid, _rank, _s = np.linalg.lstsq(np.vstack([np.ones_like(global_mean_ts),global_mean_ts]).T, d_sub.T, rcond=None)  # simple linear fit
        _cs, _ms = _model
        _pred = (global_mean_ts[:,None] * _ms + _cs).T
        d_sub = d_sub - _pred

    # subtract means from the input data
    d_sub -= np.mean(d_sub, axis=1)[:, None]

    # normalize the data
    d_sub /= np.sqrt(np.sum(d_sub * d_sub, axis=1))[:, None]

    # from:
    # https://stackoverflow.com/questions/24717513/python-np-corrcoef-memory-error
    # more efficient, potentially :-/
    numrows = d_sub.shape[0]

    if (out_dir is None) and (fname is None):
        print("If you do not have a filename input, you must set an output directory for the h5 matrix file.")
        return 0
    elif out_dir is not None:
        if fname is not None:
            h5_fname = os.path.join(out_dir,fname_head + fname.split('/')[-1].split('.')[0] + "_" + str(numrows) + "els_corr_mat.h5")
        else:
            h5_fname = os.path.join(out_dir, fname_head + str(numrows) + "els_pearson_r_corr_mat.h5")
    elif fname is not None:
        h5_fname = fname.split('.')[0] + str(numrows) + "els_corr_mat.h5"

    if os.path.exists(h5_fname) and not CLOBBER:
        print("Output file already exists, not overwriting because you didn't tell me to\n{}".format(h5_fname))
        return {'h5_fname':h5_fname}
    else:
        print("You have to chosen to write over a previously created output file.")


    #correlate this block, dump to the huge storage array
    r_idx = 0
    if n_vox_per_corr_block > d_sub.shape[0]:
        n_vox_per_corr_block = d_sub.shape[0]
    print('\nIterating processing over {} row chunks'.format(len(range(0, numrows, n_vox_per_corr_block))))
    for r in range(0, numrows, n_vox_per_corr_block):
        r_idx +=1
        st_r = time.time()
        for c in range(0, numrows, n_vox_per_corr_block):
            if c < r:
                pass #we only collect the chunks that cover the upper triangle, since they are the same, we flip the results computed below
            else:
                r1 = r + n_vox_per_corr_block
                c1 = c + n_vox_per_corr_block
                chunk1 = d_sub[r:r1,:]
                chunk2 = (d_sub[c:c1,:]).T
                # print("r: {}\t-\t{}".format(r,r1))
                # print("c: {}\t-\t{}".format(c,c1))
                if c == r:
                    res[r:r1, c:c1] = np.dot(chunk1, chunk2).astype(dtype) # we take them all
                    pred_r_res[r:r1, c:c1] = rcut_2d_predict(X_HE[r:r1, c:c1],Y_HE[r:r1, c:c1], fit_opts['popt'], fit_type=fit_type).reshape(pred_r_res[r:r1, c:c1].shape) #this is very slow
                else:
                    res[r:r1, c:c1] = np.dot(chunk1, chunk2).astype(dtype)
                    res[c:c1, r:r1] = np.flipud(np.rot90(res[r:r1, c:c1])) #we compute one and then flip to the other (to make summary easier)
                    pred_r_res[r:r1, c:c1] = rcut_2d_predict(X_HE[r:r1, c:c1],Y_HE[r:r1, c:c1], fit_opts['popt'], fit_type=fit_type).reshape(pred_r_res[r:r1, c:c1].shape) #this is fairly slow (but running it in the loops with memmapping decreases mem overhead severely)
                    pred_r_res[c:c1,r:r1] = np.flipud(np.rot90(pred_r_res[r:r1, c:c1]))


        if verbosity > 1:
            print("{}:{:.2f}s ".format(r_idx,time.time()-st_r),end=' ')
        elif verbosity > -1:
            print(r_idx,end=' ')
    print("")
    print("  Time elapsed for reading data and correlation calculation: {0:.2f}s".format(time.time() - st_orig))

    st = time.time()
#     f = open_hdf5(h5_fname)
#     if n_vox_per_corr_block < np.sum(z_mask):
#         row_chunk = n_vox_per_corr_block
#     else:
#         row_chunk = np.sum(z_mask)
    print(res.shape)
#     print("Attempting to write your data to an hdf5 file")
    #this works if you remove the masking of the array when it was setup, but then the mean calcs don't work
    #dset = write_mat_hdf5(f, res, (row_chunk,np.sum(z_mask)), dset_name="pearson_r", dtype=dtype)
#     dset = write_mat_hdf5(f, np.triu(res.astype(dtype)), (row_chunk,np.sum(z_mask)), dset_name="pearson_r", dtype=dtype)
#     print("  Time for writing compressed hdf5 file [chunks = ({1},{2})]: {0:.2f}s".format(time.time() - st,row_chunk,np.sum(z_mask)))
#     print("  {}".format(h5_fname))
    
#     st = time.time()
#     res_vec = res[np.triu_indices_from(res,k=1)] # vectorize, skipping the diagonal
#     pred_r_res_vec = pred_r_res[np.triu_indices_from(pred_r_res,k=1)]
    
    ##### alternative: go straight to computing the proportion in a mem efficient way with booleans
    ### calculates the proportion of r-values from the original data that are larger than those predicted by surrogate calcs
    r_greater_mat = np.abs(res) > pred_r_res #this can be used quickly IFF you ensure that the lower 1/2 of the matrix is also filled for pred_r_res
    #this is ordered according to the HE_vec that was input, and the 0th dimension of the input data
    r_greater_els_prop_vec = (r_greater_mat.sum(axis=0)-1)/(r_greater_mat.shape[0]-1) 
    ###
    ##### maybe this is more reasonable for output?
    print("  Time for getting indices from correlation matrix and converting to vector {0:.2f}s".format(time.time() - st))
    del res
    del pred_r_res
    return r_greater_els_prop_vec
    

    # Turn back into an img
    if img_input_output:
        res_d = np.zeros_like(z_mask).astype(np.float16)
        res_d[z_mask] = corr_out
        head['cal_min'] = np.min(corr_out)
        head['cal_max'] = np.max(corr_out)
        img_corr = nb.Nifti1Image(res_d, aff, header=head)
        try:
            corr_out_name = os.path.join(h5_fname.split('.')[0] + "_mean.nii.gz")
            img_corr.to_filename(corr_out_name)
            print("Image file written to:\n{0}\n".format(corr_out_name))
        except:
                print("Image files were not written to disk properly:\n{0}\n".format(corr_out_name))
    print("Total time elapsed: {0:.2f}s".format(time.time() - st_orig))
    if img_input_output:
        dset2 = write_mat_hdf5(f, img_corr.get_data(), None, dset_name="pearson_r_mean_img",
                              dtype=dtype)
        close_hdf5(f)
        return {'h5_fname':h5_fname,'pearson_r_mean': img_corr}
    else:
        dset2 = write_mat_hdf5(f, corr_out, None, dset_name="pearson_r_mean_vec",
                              dtype=dtype)
        close_hdf5(f)
        return {'h5_fname':h5_fname,'pearson_r_mean': corr_out, 'pearson_r_std':corr_out_std }

    
def correlate_temporally_flipped_timecourse_memmaped(img, mask=None, out_dir=None,
                                  global_mean_signal_regression=False, dtype='float16', verbosity=0,
                                  n_vox_per_corr_block = 1000,
                                  absolute_value=True, metric='mean',out_fname_head=None,
                                  CLOBBER=False):
    import tempfile
    import os

    st_orig = time.time()

    img_input_output = True

    # if by_voxel_group is None:  # if none provided, we just make the voxel group arbitrarily large
    by_voxel_group = int(10e10)
    if isinstance(img, str):
        fname = img
        img = nb.load(img)
        print("File image input/output")
    elif isinstance(img,np.ndarray):
        img_input_output = False
        fname = None
    else:
        fname = None

    if out_fname_head is not None:
        fname_head = out_fname_head + "_"
    else:
        fname_head = ""

    try:
        head = img.header  # may not need this in the end
        aff = img.affine
    except:
        if verbosity > -1:
            print("No header / affine in img input, treating as array")
    if mask is None:
        if verbosity > -1:
            print("Don't be lazy, provide a mask please.")
            print('... in the meantime, using non-zero data from first volume to create mask.')
            print('... just because I like you!\n')
        if img_input_output:
            z_mask = np.array(img.dataobj[..., 0]).astype(bool)
        else:
            z_mask = img[...,0][...,None,None].astype(bool)
    else:
        if isinstance(mask, str):
            mask = nb.load(mask)
            z_mask = mask.get_data().astype(bool)
        elif isinstance(mask, np.ndarray):
            if np.ndim(mask) >2:
                z_mask = mask.astype(bool)
            else: #or else we take the first dim, since this should be a list of timecourses
                z_mask = mask.astype(bool)[:,0]

    ## set up the computations for the blocked analyses ##
    # determine where data is in the x,y,z directions via mask
    all_nonzero = np.array(np.where(z_mask))
    min_max = np.vstack((np.min(all_nonzero, axis=1), np.max(all_nonzero, axis=1)))

    # break data into blocks based on the size and the by_voxel_group number
    # fancy indexing does not work, at all, block indexing does, but annoying
    dim_diff = np.diff(min_max, axis=0)[0] + 1
    largest_dim = np.argmax(dim_diff)  # this is the dim that we will split on
    x_idxs = min_max[:, 0]
    y_idxs = min_max[:, 1]
    z_idxs = min_max[:, 2]
    blocks = np.array_split(np.arange(dim_diff[largest_dim]),
                            np.ceil(dim_diff[largest_dim] / float(by_voxel_group)))
    block = blocks[0]

    # set our output arrays that we will fill (weeee)
    corr_out = np.zeros(z_mask.shape).astype(float)
    corr_out_flip = np.zeros(z_mask.shape).astype(float)
    # blk_counter = 0
    print("Computing summary of pair-wise correlations between all time series.")
    print("Iterating over {} block(s) of data".format(len(blocks) ** 3))
    print("Total data shape: {}".format(img.shape))
    print("Total number of voxels within the mask: {}".format(np.sum(z_mask)))

    if global_mean_signal_regression:
        if verbosity > 0:
            print("Global mean signal regression will be performed")
        global_mean_ts = compute_global_mean_signal(img,mask=z_mask)

    print(z_mask.shape)
    # try:
    tf = tempfile.NamedTemporaryFile()
    res = np.memmap(tf, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))
    res = np.ma.masked_array(res,np.eye(res.shape[0],dtype=bool)) #THIS MAY BLOWUP WITH LARGE SIZES

    #this, and the other code above, is legacy code that was used for the HE calculations and worked around here
    x_coords = block + x_idxs[0]
    y_coords = block + y_idxs[0]
    z_coords = block + z_idxs[0]

    if verbosity > 1:
        print("  x,y,z range: {}-{}, {}-{}, {}-{}".format(x_coords[0], x_coords[-1] + 1, y_coords[0],
                                                          y_coords[-1] + 1, z_coords[0], z_coords[-1] + 1))

    if img_input_output:
        d_sub = np.copy(img.get_data()[z_mask,:])
        # d_sub = np.array(img.dataobj[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1,
        #                  z_coords[0]:z_coords[-1] + 1, :])
    else: #we passed a 2d matrix, we just make sure that it doesn't have anything extra going on...
        d_sub = np.copy(np.squeeze(img))
        d_sub = d_sub[z_mask[:,0,0],:] #to only select voxels where there are no zeros in the first frame

        # if clean_timecourse_data:
        #     d_sub = _clean_timecourse(d_sub[m_sub, :])
        # else:
        #     d_sub = d_sub[m_sub, :]

    if global_mean_signal_regression: #this should do global mean signal regression, though not tested against other implementations
        _model, _resid, _rank, _s = np.linalg.lstsq(np.vstack([np.ones_like(global_mean_ts),global_mean_ts]).T, d_sub.T, rcond=None)  # simple linear fit
        _cs, _ms = _model
        _pred = (global_mean_ts[:,None] * _ms + _cs).T
        d_sub = d_sub - _pred

    # subtract means from the input data
    d_sub -= np.mean(d_sub, axis=1)[:, None]

    # normalize the data
    d_sub /= np.sqrt(np.sum(d_sub * d_sub, axis=1))[:, None]

    # from:
    # https://stackoverflow.com/questions/24717513/python-np-corrcoef-memory-error
    # more efficient, potentially :-/
    numrows = d_sub.shape[0]


    if (out_dir is None) and (fname is None):
        print("If you do not have a filename input, you must set an output directory for the h5 matrix file.")
        return 0
    elif out_dir is not None:
        if fname is not None:
            h5_fname = os.path.join(out_dir,fname_head + fname.split('/')[-1].split('.')[0] + "_" + str(numrows) + "els_tflipped_corr_mat.h5")
        else:
            h5_fname = os.path.join(out_dir, fname_head + str(numrows) + "els_pearson_r_tflip_corr_mat.h5")
    elif fname is not None:
        h5_fname = fname.split('.')[0] + str(numrows) + "els_corr_mat.h5"

    if os.path.exists(h5_fname) and not CLOBBER:
        print("Output file already exists, not overwriting because you didn't tell me to\n{}".format(h5_fname))
        return {'h5_fname':h5_fname}
    else:
        print("You have to chosen to write over a previously created output file.")

    #correlate this block, dump to the huge storage array
    r_idx = 0
    if n_vox_per_corr_block > d_sub.shape[0]:
        n_vox_per_corr_block = d_sub.shape[0]
    print('\nIterating processing over {} row chunks'.format(len(range(0, numrows, n_vox_per_corr_block))))
    for r in range(0, numrows, n_vox_per_corr_block):
        r_idx +=1
        st_r = time.time()
        for c in range(0, numrows, n_vox_per_corr_block):
            if c < r:
                pass #we only collect the chunks that cover the upper triangle, since they are the same, we flip the results computed below
            else:
                r1 = r + n_vox_per_corr_block
                c1 = c + n_vox_per_corr_block
                chunk1 = d_sub[r:r1,:]
                chunk2 = (d_sub[c:c1,:][:,::-1]).T
                # print("r: {}\t-\t{}".format(r,r1))
                # print("c: {}\t-\t{}".format(c,c1))
                if c == r:
                    res[r:r1, c:c1] = np.dot(chunk1, chunk2).astype(dtype) # we take them all
                else:
                    res[r:r1, c:c1] = np.dot(chunk1, chunk2).astype(dtype)
                    res[c:c1, r:r1] = np.flipud(np.rot90(res[r:r1, c:c1])) #we compute one and then flip to the other (to make summary easier)

        if verbosity > 1:
            print("{}:{:.2f}s ".format(r_idx,time.time()-st_r),end=' ')
        elif verbosity > -1:
            print(r_idx,end=' ')
    print("")
    print("  Time elapsed for reading data and correlation calculation: {0:.2f}s".format(time.time() - st_orig))

    st = time.time()
    f = open_hdf5(h5_fname)
    if n_vox_per_corr_block < np.sum(z_mask):
        row_chunk = n_vox_per_corr_block
    else:
        row_chunk = np.sum(z_mask)
    # print(res.shape)
    dset = write_mat_hdf5(f, np.triu(res.astype(dtype)), (row_chunk,np.sum(z_mask)), dset_name="pearson_r_tflip", dtype=dtype)
    print("  Time for writing compressed hdf5 file [chunks = ({1},{2})]: {0:.2f}s".format(time.time() - st,row_chunk,np.sum(z_mask)))
    print("  {}".format(h5_fname))

    #compute the summary metric
    if absolute_value:
        print('Using the absolute value of correlations to compute the mean.')
        res = np.abs(res)
    if metric == 'mean':
        #corr_out = (np.sum(res,axis=1)-1)/(res.shape[0]-1) #compute mean by summing, subtracting diagonal, and dividing by number
        # corr_out = np.mean(res,axis=1)
        corr_out = np.zeros(res.shape[0])
        for row_idx in np.arange(0,res.shape[0]):
            corr_out[row_idx] = np.mean(res[row_idx,:])
    corr_out_std = corr_out #temp TODO: update to loop over requested metrics

    del res
    # Turn back into an img
    if img_input_output:
        res_d = np.zeros_like(z_mask).astype(np.float16)
        res_d[z_mask] = corr_out
        head['cal_min'] = np.min(corr_out)
        head['cal_max'] = np.max(corr_out)
        img_corr = nb.Nifti1Image(res_d, aff, header=head)
        try:
            corr_out_name = os.path.join(h5_fname.split('.')[0] + "_mean.nii.gz")
            img_corr.to_filename(corr_out_name)
            print("Image file written to:\n{0}\n".format(corr_out_name))
        except:
                print("Image files were not written to disk properly:\n{0}\n".format(corr_out_name))
    print("Total time elapsed: {0:.2f}s".format(time.time() - st_orig))
    if img_input_output:
        dset2 = write_mat_hdf5(f, img_corr.get_data(), None, dset_name="pearson_r_tflip_mean_img",
                              dtype=dtype)
        close_hdf5(f)
        return {'h5_fname':h5_fname,'pearson_r_tflip_mean': img_corr}
    else:
        dset2 = write_mat_hdf5(f, corr_out, None, dset_name="pearson_r_tflip_mean_vec",
                              dtype=dtype)
        close_hdf5(f)
        return {'h5_fname':h5_fname,'pearson_r_tflip_mean': corr_out, 'pearson_r_tflip_std':corr_out_std }

def subselect_HE_corr_mat_sort(HE_vec, h5_corr_mat_fname, num_els=1000, r_mat_name='pearson_r', custom_subsample_indices=None,
                               trim_proportion=0.05):
    """
    Returns a sorted subselection of HE_vec and correlation matrix from num_els elements linearly spaced along a sorted HE_vec, after trimming the set
    trim_proportion from the the head and tail of sorted vector.

    :param HE_vec:
        HE estimates in vector form, matching with h5 matrix
    :param h5_corr_mat_fname:

    :param num_els:
        Number of elements to sample from the HE_vec and matrix
    :param r_mat_name:
        Internal dataset name for correlation matrix stored in h5 file (default = 'pearson_r')
    :param custom_subsample_indices:
        Custom set of indices to extract. Useful for looping over the entire h5 file
        If set, overrides num_els
    :param trim_proportion:
        Proportion of total number of elements to trim from head/tail of vector and matrix
    :return:
        {'HE_sort':HE_sort, 'r_mat_sub':r_mat_sub_full}
    """
    # TODO: allow passing of pointer to H5 file rather than fname, allowing faster access for reading if custom_subsample_indices is set- particularly reading does not fit with chunks?
    f = h5py.File(h5_corr_mat_fname)
    r_mat = f[r_mat_name]
    tot_num_els = r_mat.shape[0]

    if (num_els >= tot_num_els) and (custom_subsample_indices is not None):
        print('I reset the subsample element number to the number of elements, you set it too high! ({} -> {})'.format(num_els, num_els))
        num_els = tot_num_els
    if custom_subsample_indices is not None:
        print('Using custom list of indices.')
        subset_idx = custom_subsample_indices
    else:
        subset_idx = np.linspace(trim_proportion*tot_num_els,tot_num_els-1-trim_proportion*tot_num_els,num=num_els).astype(int)

    HE_sort_idx = np.argsort(HE_vec)

    ordered_sub_idx = HE_sort_idx[subset_idx]  # subselect indices from the sorted indices
    sort_ordered_sub_idx = np.sort(ordered_sub_idx)
    idx_ordered_to_HE_ordered = np.digitize(ordered_sub_idx,
                                            sort_ordered_sub_idx) - 1  # gets the mapping of the HE to index sorted (subtract one to be 0-indexed)


    # since we only saved the triu, we have to fully reconstruct before re-ordering
    r_mat_sub = r_mat[sort_ordered_sub_idx, :][:, sort_ordered_sub_idx]
    r_mat_sub_lower = np.flipud(np.rot90(np.triu(r_mat_sub, k=1)))  # construct the lower part of triangle

    r_mat_sub_full = r_mat_sub + r_mat_sub_lower #add the lower to the upper #TODO: faster with indices?
    r_mat_sub_full = r_mat_sub_full[idx_ordered_to_HE_ordered, :][:,
                     idx_ordered_to_HE_ordered]  # now we re-order it so that it is back to HE ordered
    HE_sort = HE_vec[ordered_sub_idx]  # selects the subset from the sorted data
    f.close()
    return {'HE_sort':HE_sort, 'r_mat_sub':r_mat_sub_full}

def simulate_ts_by_powerlaw(n_samples, alpha):
    '''
    Original code adapted from matlab for python by Christopher Steele (2018)

    Creates time series specified alpha-exponent, algorithm according to:
    Kasdin, N. Jeremy.
    "Discrete simulation of colored noise and stochastic processes
    and 1/f^\alpha power law noise generation."
    Proceedings of the IEEE 83.5 (1995): 802-827.
    IN:
        nr_samples [integer] : length of time series
        alpha      [integer] : time series has specified alpha-exponent
    OUT:
          X [nr_samples x 1] : time series with specified alpha-exponent

    :param n_samples:
    :param alpha:
    :return:
    '''

    beta = 2. * alpha - 1.
    Q_d = 1  # white noise will be in the range [-Q_d,Q_d]

    #  generate the coefficients h_k.
    hfa = np.zeros((2 * n_samples + 1, 1))  # add one on, to avoid division by zero in for loop
    hfa[0:2] = 1
    for i in np.arange(2, n_samples):
        hfa[i] = hfa[i - 1] * (beta / 2. + (i - 2)) / (i - 1)
    hfa = hfa[1:]

    # fill the sequence w_k with white noise and pad with zeroes
    wfa = np.vstack([-Q_d + 2 * Q_d * np.random.random_sample((n_samples, 1)), np.zeros((n_samples, 1))])

    # perform the discrete Fourier transforms
    fh = np.fft.fft(np.squeeze(hfa))
    fw = np.fft.fft(np.squeeze(wfa))

    # multiply the two complex vectors and pad with zeroes
    complex_prod = np.multiply(fh, fw)
    complex_prod = np.hstack([complex_prod[0:n_samples], np.zeros((n_samples))])

    #  inverse Fourier transform the result.
    X = np.fft.ifft(complex_prod)
    X = np.real(X[0:n_samples])
    return X

def compute_paired_surrogates(timeseries, num_surrogates=1):
    '''
    Take in time series timeseries and compute surrogate timeseries via Adjusted Amplitude Fourier Transform
    Python version of a subset of the code used to compute paired surrogates and establish significance.
    This is slow, particularly the fft, and multiply (profiled with lprun)
    '''
    # %% COMPUTE SIGNIFICANCE WITH SURROGATE DATA PERMUTATION TEST
    # % N Schaworonkow, DAJ Blythe, J Kegeles, G Curio, VV Nikulin:
    # % Power-law dynamics in neuronal and behavioral data introduce spurious
    # % correlations. Human Brain Mapping. 2015.
    # % http://doi.org/10.1002/hbm.22816

    # % Surrogates via AAFT: Adjusted Amplitude Fourier Transform
    # % James Theiler, Stephen Eubank, Andre; Longtin, Bryan Galdrikian,
    # % and J. Doyne Farmer. 1992. Testing for nonlinearity in time series:
    # % the method of surrogate data. Phys. D 58, 1-4 (September 1992), 77-94.
    # % DOI=10.1016/0167-2789(92)90102-S
    # % http://dx.doi.org/10.1016/0167-2789(92)90102-S

    nr_samples = timeseries.shape[0]
    all_surrogates = np.zeros((num_surrogates, nr_samples))

    for idx in np.arange(0, num_surrogates):
        # create white noise vector with n entries
        white_noise = np.sort(np.random.randn(nr_samples))  # use noise samples from std. norm distribution
        # sort z and extract the ranks
        ranks = np.argsort(timeseries)
        sorted_z = timeseries[ranks]
        idx_ranks = np.argsort(ranks)

        # random phase surrogate on white noise
        Z_amps = np.squeeze(np.abs(np.fft.fft(white_noise[idx_ranks])))

        rand_phases = np.squeeze(np.random.rand(1, np.floor(nr_samples / 2.).astype(int)) * 2. * np.pi)

        if np.mod(nr_samples, 2) == 0:
            start = 1  # we skip the first one, since we append with a 0 and this will be uneven
        else:
            start = 0  # or we take the whole thing

        #rand_phases = np.hstack([0, rand_phases, -rand_phases[::-1][start:]])
        rand_phases = np.append(np.append([0], rand_phases), -rand_phases[::-1][start:])
        # put amps and phases together for complex Fourier spectrum
        white_noise_rand_phase = np.multiply(Z_amps, np.exp(1j * rand_phases))
        # project the complex spectrum back to the time domain
        white_noise_rand_phase = np.real(np.fft.ifft(white_noise_rand_phase))

        # extract the ranks of the phase randomized white noise
        ranks = np.argsort(white_noise_rand_phase)
        idx_ranks = np.argsort(ranks)

        # assign ranks of phase randomized normal deviated to sorted_z
        # obtain AAFT surrogates
        z_surrogate = sorted_z[idx_ranks]

        all_surrogates[idx, :] = z_surrogate
    return all_surrogates

def compute_paired_surrogates_mat(timeseries_mat):
    '''
    Take in timeseries matrix (elementsXsamples) and compute surrogate timeseries via Adjusted Amplitude Fourier Transform
    This has been sped up to work with matrices rather than timeseries, but will only generate a single surrogate 
    for each of the timeseries in the matrix (NOTE: this behaviour is different from that above)
    
    Python version of a subset of the code used to compute paired surrogates and establish significance.
    
    This is significantly faster than looping over single computations as in compute_paired_surrogates
    '''
    # %% COMPUTE SIGNIFICANCE WITH SURROGATE DATA PERMUTATION TEST
    # % N Schaworonkow, DAJ Blythe, J Kegeles, G Curio, VV Nikulin:
    # % Power-law dynamics in neuronal and behavioral data introduce spurious
    # % correlations. Human Brain Mapping. 2015.
    # % http://doi.org/10.1002/hbm.22816

    # % Surrogates via AAFT: Adjusted Amplitude Fourier Transform
    # % James Theiler, Stephen Eubank, Andre; Longtin, Bryan Galdrikian,
    # % and J. Doyne Farmer. 1992. Testing for nonlinearity in time series:
    # % the method of surrogate data. Phys. D 58, 1-4 (September 1992), 77-94.
    # % DOI=10.1016/0167-2789(92)90102-S
    # % http://dx.doi.org/10.1016/0167-2789(92)90102-S

    nr_samples = timeseries_mat.shape[1]
    num_surrogates = timeseries_mat.shape[0]
    all_surrogates = np.zeros((num_surrogates, nr_samples))

    # TODO: could be substantially sped up with an optimised fft and ifft
    # create white noise vector with n entries
    white_noise = np.sort(np.random.randn(timeseries_mat.shape[0],nr_samples))  # use noise samples from std. norm distribution
    # sort z and extract the ranks
    ranks = np.argsort(timeseries_mat,axis=1)
    sorted_z = np.take_along_axis(timeseries_mat,ranks,axis=1) #take_along_axis is as fast or faster than equivalent .ravel() and .reshape() operations
    idx_ranks = np.argsort(ranks,axis=1)

    # random phase surrogate on white noise
    Z_amps = np.squeeze(np.abs(np.fft.fft(np.take_along_axis(white_noise,idx_ranks,axis=1),axis=1)))

    rand_phases = np.squeeze(np.random.rand(timeseries_mat.shape[0], np.floor(nr_samples / 2.).astype(int)) * 2. * np.pi)

    if np.mod(nr_samples, 2) == 0:
        start = 1  # we skip the first one, since we append with a 0 and this will be uneven
    else:
        start = 0  # or we take the whole thing

    rand_phases = np.hstack([np.hstack([np.zeros(timeseries_mat.shape[0])[...,np.newaxis],rand_phases]), -rand_phases[:,::-1][:,start:]])
    # put amps and phases together for complex Fourier spectrum
    white_noise_rand_phase = np.multiply(Z_amps, np.exp(1j * rand_phases))
    # project the complex spectrum back to the time domain
    white_noise_rand_phase = np.real(np.fft.ifft(white_noise_rand_phase,axis=1))

    # extract the ranks of the phase randomized white noise
    ranks = np.argsort(white_noise_rand_phase,axis=1)
    idx_ranks = np.argsort(ranks,axis=1)

    # assign ranks of phase randomized normal deviated to sorted_z
    # obtain AAFT surrogates
    z_surrogate = np.take_along_axis(sorted_z,idx_ranks,axis=1)

    all_surrogates = z_surrogate
    return all_surrogates

def simulate_ts_by_exponent_ce(n_samples, H):
    """
    Computes a time series of fractional brownian motion with circulant embedding. I.e., generate a time series
    based on a Hurst exponent
    adapted from matlab code available in the following publication:
    Spatial Process Generation
    Krose & Botev, 2013?
    https://arxiv.org/pdf/1308.0399v1.pdf

    :param n_samples:
    :param H: Desired hurst exponent
    :return:
    """

    # generate timeseries based on hurst
    r = np.zeros(n_samples + 2) * np.nan
    r[0:2] = 1
    for k in np.arange(1, n_samples + 1):
        r[k + 1] = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + (k - 1) ** (2 * H))
    r = r[1:]
    r = np.hstack([r, r[::-1][1:-1]])
    ld = np.real(np.fft.fft(r)) / (2 * n_samples)
    W = np.fft.fft(np.sqrt(ld) * (np.random.randn(2 * n_samples) + np.random.randn(2 * n_samples) * 1j))
    W = n_samples ** (-H) * np.cumsum(np.real(W[0:n_samples]))  # rescale
    sig = W - np.hstack([0, W[:-1]])  # remove the cumsum by shifting and subtracting
    return sig


def compute_global_mean_signal(img,mask=None,verbosity=0):
    """
    Compute the global mean signal of all elements defined in binary mask (1=select).
    #TODO: could be blocked running mean computation (sum... then divide by num elements)
    :param img:
    :param mask:
    :param verbosity:
    :return: global_mean_ts - global mean timeseries (mean of all elements in each timepoint, for each timepoint
    """
    img_input_output = True
    if isinstance(img, str):
        fname = img
        img = nb.load(img)
    elif isinstance(img,np.ndarray):
        img_input_output = False
    elif isinstance(img, nb.Nifti1Image):
        img = img
    else:
        fname = None

    try:
        head = img.header  # may not need this in the end
        aff = img.affine
    except:
        if verbosity > -1:
            print("No header / affine in img input, treating as array")
    if mask is None:
        if verbosity > -1:
            print("Don't be lazy, provide a mask please.")
            print('... in the meantime, using non-zero data from first volume to create mask.')
            print('... just because I like you!\n')
        if img_input_output:
            z_mask = np.array(img.dataobj[..., 0]).astype(bool)
        else:
            z_mask = img[...,0][...,None,None].astype(bool)
    else:
        if isinstance(mask, str):
            mask = nb.load(mask)
            z_mask = mask.get_data().astype(bool)
        elif isinstance(mask, np.ndarray):
            if np.ndim(mask) >2:
                z_mask = mask.astype(bool)
            else: #or else we take the first dim, since this should be a list of timecourses
                z_mask = mask.astype(bool)[:,0]
    z_mask = np.squeeze(z_mask)
    if img_input_output:
        global_mean_ts = np.squeeze(np.mean(img.get_data()[z_mask,:], axis=0))
    else:
        global_mean_ts = np.squeeze(np.mean(img[z_mask,:], axis=0))
    return global_mean_ts

def correlate_surrogate_timecourse_memmaped(img, actual_correlation_h5_fname=None,
                                            actual_corr_mat_dset_name = 'pearson_r', num_surrogate_reps=250, mask=None,
                                            out_dir=None, global_mean_signal_regression=False, dtype='float16',
                                            verbosity=0, n_vox_per_corr_block=1000, percentile = 95,
                                            metric='all',numba_parallel=False,out_fname_head=None, CLOBBER=False,
                                            compute_mean_sd=True):
    """
    This differs from correlate_timecourse_memmapped in that it uses the absolute value of the correlations (magnitude)
    :param img:
    :param num_surrogate_reps: stabilizes, on average, at appx 200 so we cut at 250, which provides 250 datapoints on one side of zero (due to abs)
    :param mask:
    :param out_dir:
    :param global_mean_signal_regression:
    :param dtype:
    :param verbosity:
    :param n_vox_per_corr_block:
    :param percentile:
    :param absolute_value:
    :param metric:
    :param compute_mean_sd: set to true to compute the mean and sd of the surrogate timecourse, if actual_corr_mat_dset_name not none, also computes zscore (which is based on the abs value of actual and surrogates)
            this is currently only in the hdf5 dataset file
    :return:
    """
    import tempfile
    import os

    img_input_output = True

    # if by_voxel_group is None:  # if none provided, we just make the voxel group arbitrarily large
    by_voxel_group = int(10e10)
    if isinstance(img, str):
        fname = img
        img = nb.load(img)
        print("File image input/output")
    elif isinstance(img,np.ndarray):
        img_input_output = False
        fname = None
    else:
        fname = None

    if out_fname_head is not None:
        fname_head = out_fname_head + "_"
    else:
        fname_head = ""

    try:
        head = img.header  # may not need this in the end
        aff = img.affine
    except:
        if verbosity > -1:
            print("No header / affine in img input, treating as array")
    if mask is None:
        if verbosity > -1:
            print("Don't be lazy, provide a mask please.")
            print('... in the meantime, using non-zero data from first volume to create mask.')
            print('... just because I like you!\n')
        if img_input_output:
            z_mask = np.array(img.dataobj[..., 0]).astype(bool)
        else:
            z_mask = img[...,0][...,None,None].astype(bool)
    else:
        if isinstance(mask, str):
            mask = nb.load(mask)
            z_mask = mask.get_data().astype(bool)
        elif isinstance(mask, np.ndarray):
            if np.ndim(mask) >2:
                z_mask = mask.astype(bool)
            else: #or else we take the first dim, since this should be a list of timecourses
                z_mask = mask.astype(bool)[:,0]

    ## set up the computations for the blocked analyses ##
    # determine where data is in the x,y,z directions via mask
    all_nonzero = np.array(np.where(z_mask))
    min_max = np.vstack((np.min(all_nonzero, axis=1), np.max(all_nonzero, axis=1)))

    # break data into blocks based on the size and the by_voxel_group number
    # fancy indexing does not work, at all, block indexing does, but annoying
    dim_diff = np.diff(min_max, axis=0)[0] + 1
    largest_dim = np.argmax(dim_diff)  # this is the dim that we will split on
    x_idxs = min_max[:, 0]
    y_idxs = min_max[:, 1]
    z_idxs = min_max[:, 2]
    blocks = np.array_split(np.arange(dim_diff[largest_dim]),
                            np.ceil(dim_diff[largest_dim] / float(by_voxel_group)))
    block = blocks[0]

    # set our output arrays that we will fill (weeee)
    corr_out = np.zeros(z_mask.shape).astype(float)
    corr_out_flip = np.zeros(z_mask.shape).astype(float)
    # blk_counter = 0
    print("Computing summary of pair-wise correlations between all surrogated time series.")
    print("Iterating over {} block(s) of data".format(len(blocks) ** 3))
    print("Total data shape: {}".format(img.shape))
    print("Total number of voxels within the mask: {}".format(np.sum(z_mask)))
    print("Number of surrogate repetitions to compute percentile cutoff: {}".format(num_surrogate_reps))

    if global_mean_signal_regression:
        if verbosity > 0:
            print("Global mean signal regression will be performed")
        global_mean_ts = compute_global_mean_signal(img,mask=z_mask)

    print(z_mask.shape)

    tf = tempfile.NamedTemporaryFile()
    res = np.memmap(tf, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))

    # convert to a masked array and mask out the diagonal
    # then don't need to worry about mean and std etc
    res = np.ma.masked_array(res, np.eye(res.shape[0], dtype=bool))  # THIS MAY BLOWUP WITH LARGE SIZES?

    if compute_mean_sd:
        tf3 = tempfile.NamedTemporaryFile()
        res_mean_sd = np.memmap(tf3, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask), 2)) #we don't compute any summary metrics, so we don't need to mask it as above

    if actual_correlation_h5_fname is not None:
        tf2 = tempfile.NamedTemporaryFile()
        res2 = np.memmap(tf2, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))
        res2 = np.ma.masked_array(res2, np.eye(res2.shape[0], dtype=bool))  # THIS MAY BLOWUP WITH LARGE SIZES?
        h5_corr = h5py.File(actual_correlation_h5_fname,'r')
        actual_corr_mat = h5_corr[actual_corr_mat_dset_name]

    #this, and the other code above, is legacy code that was used for the HE calculations and worked around here
    x_coords = block + x_idxs[0]
    y_coords = block + y_idxs[0]
    z_coords = block + z_idxs[0]

    if verbosity > 1:
        print("  x,y,z range: {}-{}, {}-{}, {}-{}".format(x_coords[0], x_coords[-1] + 1, y_coords[0],
                                                          y_coords[-1] + 1, z_coords[0], z_coords[-1] + 1))

    if img_input_output:
        d_sub = np.copy(img.get_data()[z_mask,:])
    else: #we passed a 2d matrix, we just make sure that it doesn't have anything extra going on...
        d_sub = np.copy(np.squeeze(img))
        d_sub = d_sub[z_mask[:,0,0],:] #to only select voxels where there are no zeros in the first frame

    numrows = d_sub.shape[0]

    if (out_dir is None) and (fname is None):
        print("If you do not have a filename input, you must set an output directory for the h5 matrix file.")
        return 0
    elif out_dir is not None:
        if fname is not None:
            h5_fname = os.path.join(out_dir,fname_head + fname.split('/')[-1].split('.')[0] + "_" + str(numrows) + "els_" + str(num_surrogate_reps) + "reps_surrogate_pearson_r_cutoff_mat.h5")
        else:
            h5_fname = os.path.join(out_dir, fname_head + str(numrows) + "els_" + str(num_surrogate_reps) + "reps_surrogate_pearson_r_cutoff_mat.h5")
    elif fname is not None:
        h5_fname = fname.split('.')[0] + "_" + str(numrows) + "els_" + str(num_surrogate_reps) + "reps_surrogate_pearson_r_cutoff_mat.h5"

    if os.path.exists(h5_fname) and not CLOBBER:
        print("Output file already exists, not overwriting because you didn't tell me to\n{}".format(h5_fname))
        return {'h5_fname':h5_fname}
    else:
        print("You have to chosen to write over a previously created output file.")

    d_sub_orig = np.copy(d_sub)
    st_orig = time.time()


    if global_mean_signal_regression:  # this should do global mean signal regression, though not tested against other implementations
        _model, _resid, _rank, _s = np.linalg.lstsq(np.vstack([np.ones_like(global_mean_ts), global_mean_ts]).T,
                                                    d_sub.T, rcond=None)  # simple linear fit
        _cs, _ms = _model
        _pred = (global_mean_ts[:, None] * _ms + _cs).T
        d_sub = d_sub - _pred

    # from:
    # https://stackoverflow.com/questions/24717513/python-np-corrcoef-memory-error
    # more efficient, potentially :-/

    # subtract means from the input data
    d_sub -= np.mean(d_sub, axis=1)[:, None]

    # normalize the data
    d_sub /= np.sqrt(np.sum(d_sub * d_sub, axis=1))[:, None]

    ## perform correlations
    #correlate this block, dump to the huge storage array
    r_idx = 0
    if n_vox_per_corr_block > d_sub.shape[0]:
        n_vox_per_corr_block = d_sub.shape[0]
        print('Adjusting block size to maximum possible with your data')
    print('\nIterating processing over {} row chunks'.format(len(range(0, numrows, n_vox_per_corr_block))))
    for r in range(0, numrows, n_vox_per_corr_block):
        r_idx +=1
        st_r = time.time()
        for c in range(0, numrows, n_vox_per_corr_block):
            if c < r:
                pass #we only collect the chunks that cover the upper triangle, since they are the same, we flip the results computed below
            else:
                r1 = r + n_vox_per_corr_block
                c1 = c + n_vox_per_corr_block
                # create a zero'd array of values to store the correlations from the surrogates in order to calculate the distribution

                if not numba_parallel:
                    _dm1 = np.zeros((d_sub[r:r1].shape[0],d_sub[c:c1].shape[0],num_surrogate_reps)).astype(dtype) #correct for missing voxels when 2d matrix passed
                    if verbosity > 0:
                        print('    Performing surrogate repetitions for this block')

                    for surr_idx in range(num_surrogate_reps):
                        if verbosity >1:
                            if surr_idx == 0:
                                print('    ', end='')
                            print('{}'.format(surr_idx+1),end=',')

                        #grab the original data before computing the surrogates
                        chunk1 = np.copy(d_sub[r:r1])
                        chunk2 = np.copy(d_sub[c:c1])

                        #calculate the surrogates, surrogates currently re-calculated for new chunks - potentially not optimal but it is not possible to do all data if surrogates created in the outer
                        # loop due to memory constraints
                        for rr in range(chunk1.shape[0]):
                            chunk1[rr,:] = compute_paired_surrogates(chunk1[rr,:], num_surrogates=1)
                        if c==r: #if we are in a diagonal, no need to calculate the surrogates again
                            chunk2 = chunk1
                        else:
                            for rr in range(chunk2.shape[0]):
                                chunk2[rr,:] = compute_paired_surrogates(chunk2[rr,:], num_surrogates=1)
                        _dm1[:,:,surr_idx] = np.abs(np.dot(chunk1, chunk2.T).astype(dtype)) #compute the data matrices (_dm), using abs because it is the same on both sides (.95, or .975 if two-sided)
                else:
                    print('>> =============================================== <<')
                    print('>> Implementing parallel processing with numba.jit <<')
                    print('>> =============================================== <<')
                    _dm1 = parallel_surr_corr(d_sub, r, r1, c, c1, dtype, num_surrogate_reps)
                if verbosity > 1:
                    print("")
                if c == r:
                    res[r:r1, c:c1] = np.percentile(_dm1, percentile, axis=-1) #calc percentile cutoff for this set of correlation distributions and place in res
                    if actual_correlation_h5_fname is not None:
                        res2[r:r1, c:c1] = (((_dm1 - actual_corr_mat[r:r1, c:c1][..., None]) < 0).sum(axis=-1))/_dm1.shape[-1] #calc the proportion of surrogate values below the true value of corr
                        res2[r:r1, c:c1] = (((_dm1 - actual_corr_mat[r:r1, c:c1][..., None]) < 0).sum(axis=-1)) /_dm1.shape[-1]  # calc the proportion of surrogate values below the true value of corr

                    if compute_mean_sd:
                        res_mean_sd[r:r1, c:c1, 0] = np.mean(_dm1, axis=-1)
                        res_mean_sd[r:r1, c:c1, 1] = np.std(_dm1, axis=-1)

                else:
                    pct_res = np.percentile(_dm1, percentile, axis=-1)
                    res[r:r1, c:c1] = pct_res
                    res[c:c1, r:r1] = np.flipud(np.rot90(pct_res))
                    if actual_correlation_h5_fname is not None:
                        prop_actual = (((_dm1 - actual_corr_mat[r:r1, c:c1][..., None]) < 0).sum(axis=-1)) / _dm1.shape[-1]
                        res2[r:r1, c:c1] = prop_actual
                        res2[c:c1, r:r1] = np.flipud(np.rot90(prop_actual))
                    if compute_mean_sd:
                        tmp_var = np.mean(_dm1, axis=-1)
                        res_mean_sd[r:r1, c:c1, 0] = tmp_var
                        res_mean_sd[c:c1, r:r1, 0] = np.flipud(np.rot90(tmp_var))
                        tmp_var = np.std(_dm1, axis=-1)
                        res_mean_sd[r:r1, c:c1, 1] = tmp_var
                        res_mean_sd[c:c1, r:r1, 0] = np.flipud(np.rot90(tmp_var))
        if verbosity > 1:
            print("{}:{:.2f}s ".format(r_idx,time.time()-st_r),end='\n')
        elif verbosity > -1:
            print(r_idx,end=' ')
    if verbosity >-1:
        print("  Time elapsed for reading data, surrogate generation, correlation, and percentile calculation: {0:.2f}s".format(time.time() - st_orig))

    st = time.time()
    f = open_hdf5(h5_fname)
    if n_vox_per_corr_block < np.sum(z_mask):
        row_chunk = n_vox_per_corr_block
    else:
        row_chunk = np.sum(z_mask)
    dset = write_mat_hdf5(f, np.triu(res.astype(dtype)), (row_chunk,np.sum(z_mask)), dset_name="pearson_r_cutoff", dtype=dtype)
    if actual_correlation_h5_fname is not None:
        dset2 = write_mat_hdf5(f, np.triu(res2.astype(dtype)), (row_chunk, np.sum(z_mask)), dset_name="prop_surr_below_actual_corr",
                              dtype=dtype)
        if compute_mean_sd: #we write the zscore for the real r vs the null r's computed through surrogate procedure
            dset3 = write_mat_hdf5(f, np.triu(((np.abs(actual_corr_mat[:,:])-res_mean_sd[:,:,0])/res_mean_sd[:,:,1]).astype(dtype)), (row_chunk, np.sum(z_mask)), dset_name="actual_r_zscore",
                              dtype=dtype)
    if compute_mean_sd: #or else we just dump the mean and sd to separate datasets
        dset3 = write_mat_hdf5(f, np.triu(res_mean_sd[:, :, 0].astype(dtype)), (row_chunk, np.sum(z_mask)), dset_name="surrogate_r_mean",
                              dtype=dtype)
        dset4 = write_mat_hdf5(f, np.triu(res_mean_sd[:, :, 1].astype(dtype)), (row_chunk, np.sum(z_mask)),dset_name="surrogate_r_sd",
                               dtype=dtype)
    print("  Time for writing compressed hdf5 file [chunks = ({1},{2})]: {0:.2f}s".format(time.time() - st,row_chunk,np.sum(z_mask)))
    print("  {}".format(h5_fname))

    # dset_temp = write_mat_hdf5(f, _dm1,_dm1.shape,
    #                        dset_name="_dm1_comparison-wise_distribution_data",
    #                        dtype=dtype)

    #compute the summary metric (performed over masked array, so we don't need to worry about the diagonals)
    if metric == 'all':
        metrics = ['mean','std','max']
    else:
        if isinstance(metric,str):
            metrics = [metric]
        else:
            metrics=metric

    res_dict = {'h5_fname':h5_fname}
    for metric in metrics:
        if metric == 'mean':
            corr_out = np.mean(res,axis=1).data #bring back to std np array (diagonal already removed by operation)
            #corr_out = (np.sum(res,axis=1)-1)/(res.shape[0]-1) #compute mean by summing, subtracting diagonal, and dividing by number
        if metric == 'std':
            corr_out = np.std(res,axis=1).data #bring back to std np array
        if metric == 'max':
            corr_out = np.max(res,axis=1).data #bring back to std np array

        # Turn back into an img
        if img_input_output:
            res_d = np.zeros_like(z_mask).astype(np.float16)
            res_d[z_mask] = corr_out
            #print(corr_out.shape)
            #print(z_mask.sum())
            head['cal_min'] = np.min(corr_out)
            head['cal_max'] = np.max(corr_out)
            img_corr = nb.Nifti1Image(res_d, aff, header=head)
            try:
                corr_out_name = os.path.join(h5_fname.split('.')[0] + "_surrogate_{}.nii.gz".format(metric))
                img_corr.to_filename(corr_out_name)
                print("Image file written to:\n{0}\n".format(corr_out_name))
            except:
                    print("Image files were not written to disk properly:\n{0}\n".format(corr_out_name))
        print("Total time elapsed: {0:.2f}s".format(time.time() - st_orig))
        if img_input_output:
            dset2 = write_mat_hdf5(f, img_corr.get_data(), None, dset_name="r_cutoff_{}_img".format(metric),
                                  dtype=dtype)

            res_dict['r_cutoff_{}_img'.format(metric)]= img_corr
        else:
            dset2 = write_mat_hdf5(f, corr_out, None, dset_name="r_cutoff_{}_vec".format(metric),
                                  dtype=dtype)
            res_dict['r_cutoff_{}_vec'.format(metric)]= corr_out
    del res

    if actual_correlation_h5_fname is not None:
        for metric in metrics:
            if metric == 'mean':
                corr_out = np.mean(res2,
                                   axis=1).data  # bring back to std np array (diagonal already removed by operation)
                # corr_out = (np.sum(res,axis=1)-1)/(res.shape[0]-1) #compute mean by summing, subtracting diagonal, and dividing by number
            if metric == 'std':
                corr_out = np.std(res2, axis=1).data  # bring back to std np array
            if metric == 'max': #Max probably not useful in this case, since it is across all correlations it will almost certainly = 1
                corr_out = np.max(res2, axis=1).data  # bring back to std np array

            # Turn back into an img
            if img_input_output:
                res_d = np.zeros_like(z_mask).astype(np.float16)
                res_d[z_mask] = corr_out
                # print(corr_out.shape)
                # print(z_mask.sum())
                head['cal_min'] = np.min(corr_out)
                head['cal_max'] = np.max(corr_out)
                img_corr = nb.Nifti1Image(res_d, aff, header=head)
                try:
                    corr_out_name = os.path.join(h5_fname.split('.')[0] + "_surrogate_prop_below_actual_{}.nii.gz".format(metric))
                    img_corr.to_filename(corr_out_name)
                    print("Image file written to:\n{0}\n".format(corr_out_name))
                except:
                    print("Image files were not written to disk properly:\n{0}\n".format(corr_out_name))
            print("Total time elapsed: {0:.2f}s".format(time.time() - st_orig))
            if img_input_output:
                dset2 = write_mat_hdf5(f, img_corr.get_data(), None, dset_name="r_prop_below_actual_{}_img".format(metric),
                                       dtype=dtype)

                res_dict['r_prop_below_actual_{}_img'.format(metric)] = img_corr
            else:
                dset2 = write_mat_hdf5(f, corr_out, None, dset_name="r_prop_below_actual_{}_vec".format(metric),
                                       dtype=dtype)
                res_dict['r_prop_below_actual_{}_vec'.format(metric)] = corr_out
        del res2

    close_hdf5(f)
    if actual_correlation_h5_fname is not None:
        close_hdf5(h5_corr)

    return res_dict


def correlate_surrogate_timecourse_binned(d_sub, n_bins=100, n_bins_r_cut=100, actual_correlation_h5_fname=None, mask=None,
                                               actual_corr_mat_dset_name = 'pearson_r', num_surrogate_reps=250,
                                               out_dir=None, global_mean_signal_regression=False, dtype='float16',
                                               verbosity=0, n_vox_per_corr_block=1000, percentile = 95.5,
                                               metric='all',out_fname_head=None, CLOBBER=False,
                                               compute_mean_sd=True,
                                               dfa_min_samples=10, dfa_max_samples=125, dfa_num_window_steps=10,
                                          remove_nans=True, retain_original_HE_range=True):
    """
    Only works with timecourse data of form elementXtimepoint (rowXcol). n_bins = 100 and n_bins_r_cut = 100 with num_surrogate_reps=125 should provide adequate sampling for must use cases.
    Doubling the num_surrogate_reps will allow appx 2x the resolution for n_bins_r_cut
        - Data covering the HE range of 0.5 - 1 requires appx n_bins=100 and num_surrogate_reps=125 to give adequate coverate of the n_bins_r_cut=100 HExHE space after the removal of grids
          where there are less than 125 surrotagate_r values for histogram computation
        - The same works for 0.5-1.5 (Data covering the HE range of 0.5 - 1.5 requires appx n_bins=100 and num_surrogate_reps=125 to give adequate coverate of the n_bins_r_cut=100 HExHE space after the
          removal of grids where there are less than 125 surrotagate_r values for histogram computation)

    :param d_sub:
    :param n_bins: defines the number of bins of HE to generate surrogates for, where the total number is n_bins*n_bins for all combinations
                    - 30-50 seems like a good rule of thumb to cover the HE grid space, but of course your mileage will vary if your HE values have a large range
                    - NOTE: computing the r_cut value based on a grid of this size will bias the r_cut values to lower numbers than expected as HE (*HE) increases due 
                      to the flattening of the correlation distribution as HE (*HE) increases. Practically, this means that binning the data for
                      r_cut calculations should be performed at a higher bin resolution, which *should* be based on both the original n_bins and the num_surrogate_reps
                      that will be filling each of the bins (of course, this assumes that re-binning based on the computed HE values results in approximately the same 
                      distribution, which is not correct as HE (*HE) increases). 100-200 bins are likely required, with more being better but dependent on more
                      data generated by more surrogate_reps and requiring more compuation time (for surrogates and for binning!) and memory.
                    - Currently, we take the extreme case (HE1=1.0; HE2=1.0) and simulate how many data points we need to accurately assess the r_cut. Using an imperfect
                      set of timeseries generated with HE=1.0 (i.e., which will be in a distribution around HE=1.0) we can simulate that between 500-750 samples are required
                      to get a relatively accurate estimate of r_cut (this value still undershoots by a small amount on the order of appx .02 HE), and that the inflection
                      point for the standard deviation of randomly sampled values from the simulated r distribution is also around this number of samples.
                    - However, the current function also re-bins resulting correlations according to the HE values that are re-computed from the surrogate timecourses, so we will 
                      be more accurate with fewer samples 
    :param n_bins_r_cut: sets the number of bins to resample data into for r_cut caclulations. Recommended minimum = 100, but 200 is better!
                            - the resulting r_cut accuracy depeds on n_bins, n_bins_r_cut, HE range of the data, and num_surrogate_reps
                            - see n_bins for brief for discussion
                            
    :param actual_correlation_h5_fname:
    :param mask:
    :param actual_corr_mat_dset_name:
    :param num_surrogate_reps:
    :param out_dir:
    :param global_mean_signal_regression:
    :param dtype:
    :param verbosity:
    :param n_vox_per_corr_block:
    :param percentile: Value to calculate r-cut threshold for each bin of data, the true value of (per-bin) false positives will be > 1-percentile with finite data.
    :                   - this can be either a single value (int or float) or a list of values. If a list, the output will be a dictionary with fields of the form: 'percentile_<VAL>'
                        - if a list and an output file is selected, ONLY the first percentile is used
                        - Unfortunately, due to the greater variance at larger HE there is still a slight bias towards more false positives at higher HE values for simulated data
                        - you could set it more stringent than the "accepted" 1-0.05
                            - when paired with the poly4_symm fitting function, over all bins, there is an appx .5% bias 
                            - such that a value of 97.5 roughly corresponds to 3% false positives, 95 to 5.5%
    :param metric:
    :param out_fname_head:
    :param CLOBBER:
    :param compute_mean_sd:
    :param dfa_min_samples:
    :param dfa_max_samples:
    :param dfa_num_window_steps:
    :param remove_nans:
    :param retain_original_HE_range: clips data to stay within the HE range of the input. This is necessary because surrogate timeseries have a distribution around the given timeseries. Setting True 
                                        just removes any datapoints that fall outside of this range
    :return:
    """
    from scipy.stats import binned_statistic_2d
    import tempfile
    import os

    img_input_output = True

    # if by_voxel_group is None:  # if none provided, we just make the voxel group arbitrarily large
    by_voxel_group = int(10e10)

    fname=None

    if out_fname_head is not None:
        fname_head = out_fname_head + "_"
    else:
        fname_head = ""

    if mask is not None:
        z_mask = mask.astype(bool)
    else:
        z_mask = d_sub[...,0][...,None,None].astype(bool) #assumes that no element will ever be == 0 exactly.

    ## set up the computations for the blocked analyses ##
    # determine where data is in the x,y,z directions via mask
    all_nonzero = np.array(np.where(z_mask))
    min_max = np.vstack((np.min(all_nonzero, axis=1), np.max(all_nonzero, axis=1)))

    # break data into blocks based on the size and the by_voxel_group number
    # fancy indexing does not work, at all, block indexing does, but annoying
    dim_diff = np.diff(min_max, axis=0)[0] + 1
    largest_dim = np.argmax(dim_diff)  # this is the dim that we will split on
    x_idxs = min_max[:, 0]
    y_idxs = min_max[:, 1]
    z_idxs = min_max[:, 2]
    blocks = np.array_split(np.arange(dim_diff[largest_dim]),
                            np.ceil(dim_diff[largest_dim] / float(by_voxel_group)))
    block = blocks[0]

    # set our output arrays that we will fill (weeee)
    corr_out = np.zeros(z_mask.shape).astype(float)

    if global_mean_signal_regression:
        if verbosity > 0:
            print("Global mean signal regression will be performed")
        global_mean_ts = compute_global_mean_signal(d_sub,mask=z_mask)

    print(z_mask.shape)

    if np.sum(z_mask) < n_bins:
        print('You have chosen too many HE bins ({}) for the size of your data ({} elements), please use a larger subset of data or fewer bins.'.format(n_bins,np.sum(z_mask)))

    #TODO: add check code to suggest and/or compute reasonable n_bins and n_bins_r_cut and num_surrogate_reps, because this should be a function of the HE range

    tf = tempfile.NamedTemporaryFile()
    res = np.memmap(tf, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))

    # convert to a masked array and mask out the diagonal
    # then don't need to worry about mean and std etc
    res = np.ma.masked_array(res, np.eye(res.shape[0], dtype=bool))  # THIS MAY BLOWUP WITH LARGE SIZES?

    if compute_mean_sd:
        tf3 = tempfile.NamedTemporaryFile()
        res_mean_sd = np.memmap(tf3, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask), 2)) #we don't compute any summary metrics, so we don't need to mask it as above

    if actual_correlation_h5_fname is not None:
        tf2 = tempfile.NamedTemporaryFile()
        res2 = np.memmap(tf2, dtype, mode='w+', shape=(np.sum(z_mask), np.sum(z_mask)))
        res2 = np.ma.masked_array(res2, np.eye(res2.shape[0], dtype=bool))  # THIS MAY BLOWUP WITH LARGE SIZES?
        h5_corr = h5py.File(actual_correlation_h5_fname,'r')
        actual_corr_mat = h5_corr[actual_corr_mat_dset_name]

    #this, and the other code above, is legacy code that was used for the HE calculations and worked around here
    x_coords = block + x_idxs[0]
    y_coords = block + y_idxs[0]
    z_coords = block + z_idxs[0]

    if verbosity > 1:
        print("Input data: x,y,z range: {}-{}, {}-{}, {}-{}".format(x_coords[0], x_coords[-1] + 1, y_coords[0],
                                                          y_coords[-1] + 1, z_coords[0], z_coords[-1] + 1))

    d_sub = d_sub[z_mask[:,0,0],:] #to only select voxels where there are no zeros in the first frame

    #1) compute DFA on input data subset
    #this DOES NOT determine if the data is any good or not, so that should be done outside of this function
    HE_vec = dfa_blocked(d_sub,min_samples=dfa_min_samples,max_samples=dfa_max_samples,
                         num_window_steps=dfa_num_window_steps,verbosity=-2,by_voxel_group=by_voxel_group)['HE']
    HE_min = HE_vec.min()
    HE_max = HE_vec.max()
    print('HE range of input data: {:.2f} - {:.2f}'.format(HE_min,HE_max))
          
    #2) determine what the sampling grid (HE1,HE2) will look like based on the min and max of the dataset
    #this is a linear sampling, but it may not be as appropriate when HE values exceed 1
    HE_subsamp = np.linspace(HE_vec.min(), HE_vec.max(), num=n_bins)  # linear is ok as long as we generate enough surrogates within each bin
    HE_m1_idx, HE_m2_idx = np.meshgrid(HE_subsamp, HE_subsamp)  # all combinations of HEs from the subsampling

    # we use a masked array to preserve the indices of the elements that we are grabbing, but grab UNIQUE elements for m1 and m2 vectors
    _t_mask = np.zeros(HE_vec.shape[0], dtype=bool)
    ma_HE_vec = np.ma.masked_array(HE_vec, _t_mask)

    m1_m2_idxs = []
    for val in np.concatenate((HE_subsamp, HE_subsamp)):
        idx = np.abs((ma_HE_vec - val)).argmin()
        _t_mask[idx] = True
        m1_m2_idxs.append(idx)

    m1_m2_idxs = np.array(m1_m2_idxs)
    print("{}x{} HE values selected to compute the sampling grid".format(np.shape(m1_m2_idxs)[0],np.shape(m1_m2_idxs)[0]))

    #select those elements that have the closest HE values from the original dataset, this is what we will use to generate surrogates
    d_sub = d_sub[m1_m2_idxs,:]
    numrows = d_sub.shape[0]

    # if (out_dir is None) and (fname is None):
    #     print("If you do not have a filename input, you must set an output directory for the h5 matrix file.")
    #     return 0
    if out_dir is not None:
        if fname is not None:
            h5_fname = os.path.join(out_dir,fname_head + fname.split('/')[-1].split('.')[0] + "_" + str(numrows) + "els_" + str(num_surrogate_reps) + "reps_surrogate_pearson_r_cutoff_mat.h5")
        else:
            h5_fname = os.path.join(out_dir, fname_head + str(numrows) + "els_" + str(num_surrogate_reps) + "reps_surrogate_pearson_r_cutoff_mat.h5")
    elif fname is not None:
        h5_fname = fname.split('.')[0] + "_" + str(numrows) + "els_" + str(num_surrogate_reps) + "reps_surrogate_pearson_r_cutoff_mat.h5"
    else:
        h5_fname = 'XXX_temp'
    if os.path.exists(h5_fname) and not CLOBBER:
        print("Output file already exists, not overwriting because you didn't tell me to\n{}".format(h5_fname))
        return {'h5_fname':h5_fname}
    else:
        print("You have to chosen to write over a previously created output file.")

    d_sub_orig = np.copy(d_sub)
    st_orig = time.time()


    if global_mean_signal_regression:  # this should do global mean signal regression, though not tested against other implementations
        _model, _resid, _rank, _s = np.linalg.lstsq(np.vstack([np.ones_like(global_mean_ts), global_mean_ts]).T,
                                                    d_sub.T, rcond=None)  # simple linear fit
        _cs, _ms = _model
        _pred = (global_mean_ts[:, None] * _ms + _cs).T
        d_sub = d_sub - _pred

    # from:
    # https://stackoverflow.com/questions/24717513/python-np-corrcoef-memory-error
    # more efficient, potentially :-/

    # subtract means from the input data
    d_sub -= np.mean(d_sub, axis=1)[:, None]

    # normalize the data
    d_sub /= np.sqrt(np.sum(d_sub * d_sub, axis=1))[:, None]

    print("Computing summary of pair-wise correlations between all surrogated time series.")
    print("Iterating over {} block(s) of data".format(len(blocks) ** 3))
    print("Total data shape: {}".format(d_sub.shape))
    print("Total number of voxels within the mask: {}".format(np.sum(z_mask)))
    print("Number of surrogate repetitions to compute percentile cutoff: {}".format(num_surrogate_reps))

    ## perform correlations
    #correlate this block, dump to the huge storage array
    r_idx = 0
    if n_vox_per_corr_block > d_sub.shape[0]:
        n_vox_per_corr_block = d_sub.shape[0]
        print('Adjusting block size to maximum possible with your data')
    print('\nIterating processing over {} row chunks'.format(len(range(0, numrows, n_vox_per_corr_block))))

    surr_HE1 = np.zeros((numrows**2)*num_surrogate_reps)*np.nan
    surr_HE2 = np.zeros((numrows**2)*num_surrogate_reps)*np.nan
    surr_r = np.zeros((numrows**2)*num_surrogate_reps)*np.nan
    surr_r_pctl_cut = np.zeros((numrows**2))*np.nan #for testing purposes

    for r in range(0, numrows, n_vox_per_corr_block):
        r_idx +=1
        st_r = time.time()
        for c in range(0, numrows, n_vox_per_corr_block):
            if c < r:
                pass #we only collect the chunks that cover the upper triangle, since they are the same, we flip the results computed below
            else:
                r1 = r + n_vox_per_corr_block
                c1 = c + n_vox_per_corr_block
                # create a zero'd array of values to store the correlations from the surrogates in order to calculate the distribution

                _dm1 = np.zeros((d_sub[r:r1].shape[0],d_sub[c:c1].shape[0],num_surrogate_reps)).astype(dtype) #correct for missing voxels when 2d matrix passed
                if verbosity > 0:
                    print('    Performing surrogate repetitions for this block')

                for surr_idx in range(num_surrogate_reps):
                    if verbosity >1:
                        if surr_idx == 0:
                            print('    ', end='')
                        print('{}'.format(surr_idx+1),end=',', flush=True)

                    #grab the original data before computing the surrogates
                    chunk1 = np.copy(d_sub[r:r1])
                    chunk2 = np.copy(d_sub[c:c1])

                    #calculate the surrogates, surrogates currently re-calculated for new chunks - potentially not optimal but it is not possible to do all data if surrogates created in the outer
                    # loop due to memory constraints
                    chunk1 = compute_paired_surrogates_mat(chunk1)
                    if c==r: #if we are in a diagonal, no need to calculate the surrogates again
                        chunk2 = chunk1
                    chunk2 = compute_paired_surrogates_mat(chunk2)
                    
                    ## legacy, this does the same thing, but a bit slower
                    #for rr in range(chunk1.shape[0]):
                        #chunk1[rr,:] = compute_paired_surrogates(chunk1[rr,:], num_surrogates=1)
                    #if c==r: #if we are in a diagonal, no need to calculate the surrogates again
                        #chunk2 = chunk1
                    #else:
                        #for rr in range(chunk2.shape[0]):
                            #chunk2[rr,:] = compute_paired_surrogates(chunk2[rr,:], num_surrogates=1)

                    # compute the corr of data matrices (_dm), using abs because it is the same on both sides
                    chunk_corr = np.abs(np.dot(chunk1, chunk2.T).astype(dtype))
                    _dm1[:,:,surr_idx] = chunk_corr

                    #3) compute the HE for each surrogate of each chunk, and record with the correlation values
                    HE1 = dfa_blocked(chunk1, min_samples=dfa_min_samples, max_samples=dfa_max_samples,
                                      num_window_steps=dfa_num_window_steps, by_voxel_group=by_voxel_group,
                                      verbosity=-1)['HE']
                    HE2 = dfa_blocked(chunk2, min_samples=dfa_min_samples, max_samples=dfa_max_samples,
                                      num_window_steps=dfa_num_window_steps, by_voxel_group=by_voxel_group,
                                      verbosity=-1)['HE']

                    #turn into meshgrid to keep track of the location of all HE's to correspond with corrs
                    _XX_HE1, _YY_HE2 = np.meshgrid(HE1, HE2)

                    #dump into a vector, with correlations as well
                    #get the index of the first nan value, so you know where to fill from
                    first_nan_idx = np.where(np.isnan(surr_HE1))[0][0]
                    v_len = _XX_HE1.ravel().shape[0]
                    # print(d_sub.shape)
                    # print(chunk1.shape)
                    # print(chunk2.shape)
                    # print(v_len)
                    # print(_XX_HE1.ravel().shape)
                    # print(surr_HE1[first_nan_idx:first_nan_idx + v_len].shape)
                    surr_HE1[first_nan_idx:first_nan_idx + v_len] = _XX_HE1.ravel()
                    surr_HE2[first_nan_idx:first_nan_idx + v_len] = _YY_HE2.ravel()
                    surr_r[first_nan_idx:first_nan_idx + v_len] = chunk_corr.ravel()

                #TODO: some of this is LEGACY CODE that can be removed 
                if verbosity > 1:
                    print("")
                if c == r:
                    if isinstance(percentile,int) or isinstance(percentile,float): #i.e., it is not a list
                        _percentile = percentile
                    else:
                        _percentile = percentile[0] #take the first one only
                    pct_res = np.percentile(_dm1, _percentile, axis=-1) #calc percentile cutoff for this set of correlation distributions and place in res
                    res[r:r1, c:c1] = pct_res
                    if actual_correlation_h5_fname is not None:
                        res2[r:r1, c:c1] = (((_dm1 - actual_corr_mat[r:r1, c:c1][..., None]) < 0).sum(axis=-1))/_dm1.shape[-1] #calc the proportion of surrogate values below the true value of corr
                        res2[r:r1, c:c1] = (((_dm1 - actual_corr_mat[r:r1, c:c1][..., None]) < 0).sum(axis=-1)) /_dm1.shape[-1]  # calc the proportion of surrogate values below the true value of corr

                    if compute_mean_sd:
                        res_mean_sd[r:r1, c:c1, 0] = np.mean(_dm1, axis=-1)
                        res_mean_sd[r:r1, c:c1, 1] = np.std(_dm1, axis=-1)

                else:
                    if isinstance(percentile,int) or isinstance(percentile,float): #i.e., it is not a list
                        _percentile = percentile
                    else:
                        _percentile = percentile[0] #take the first one only
                    pct_res = np.percentile(_dm1, _percentile, axis=-1)
                    res[r:r1, c:c1] = pct_res
                    res[c:c1, r:r1] = np.flipud(np.rot90(pct_res))
                    if actual_correlation_h5_fname is not None:
                        prop_actual = (((_dm1 - actual_corr_mat[r:r1, c:c1][..., None]) < 0).sum(axis=-1)) / _dm1.shape[-1]
                        res2[r:r1, c:c1] = prop_actual
                        res2[c:c1, r:r1] = np.flipud(np.rot90(prop_actual))
                    # if compute_mean_sd:
                    #     tmp_var = np.mean(_dm1, axis=-1)
                    #     res_mean_sd[r:r1, c:c1, 0] = tmp_var
                    #     res_mean_sd[c:c1, r:r1, 0] = np.flipud(np.rot90(tmp_var))
                    #     tmp_var = np.std(_dm1, axis=-1)
                    #     res_mean_sd[r:r1, c:c1, 1] = tmp_var
                    #     res_mean_sd[c:c1, r:r1, 0] = np.flipud(np.rot90(tmp_var))

                first_nan_idx = np.where(np.isnan(surr_r_pctl_cut))[0][0]
                surr_r_pctl_cut[first_nan_idx:first_nan_idx+chunk1.shape[0]*chunk2.shape[0]] = pct_res.ravel() #also convert each comparison's percentile cut into vector
        if verbosity > 1:
            print("\n  {}: {:.2f}s ".format(r_idx,time.time()-st_r),end='\n')
        elif verbosity > -1:
            print(r_idx,end=' ')

    print('\nBinning and computing percentile cut for surrogate r-scores.', end='\n')
    print('HE and r-score vectors of shape: {}'.format(len(surr_HE1)))

    #4) since the computed surrogates can have HEs that differ from the timeseries used for input, and this value becomes less similar as HE increases,
    #   we rebin the data into a 2d grid based on the actual HE1 and HE2 values prior to computing the r_cut values
    #   this could balloon memory if a large grid is used...
    # Note: Larger HEs are less well recreated by surrogate procedure (>1)

    # create boolean lookup of self correlations (r==1) and remove it
    self_corr_mask = surr_r == 1.0
    
    # create booleans of the values greater or less than the original HE range
    if retain_original_HE_range:
        below_mask = np.logical_or(surr_HE1 < HE_min,surr_HE2 < HE_min)
        above_mask = np.logical_or(surr_HE1 > HE_max,surr_HE2 > HE_max)
        removal_mask = np.logical_or(below_mask,above_mask,self_corr_mask)
    else:
        removal_mask = self_corr_mask
    surr_HE1 = surr_HE1[~removal_mask]
    surr_HE2 = surr_HE2[~removal_mask]
    surr_r = surr_r[~removal_mask]

    d_max = np.max(np.append(surr_HE1, surr_HE2))
    d_min = np.min(np.append(surr_HE1, surr_HE2))

    #calculate binned values for output -- this is very slow for the cusom get_pctl call for n_bins_r_cut > 100
    # can be done on a list, or on a single int
    if isinstance(percentile,int) or isinstance(percentile,float):
        #function for binned_statistic
        def get_pctl(vec,pctl=percentile):
            return np.percentile(vec,pctl)
        surr_r_bin_pctl = binned_statistic_2d(surr_HE1,surr_HE2,surr_r,statistic=get_pctl,bins=n_bins_r_cut).statistic.ravel()
    else: #assume a list or numpy array
        surr_r_bin_pctl = {}
        print("Computing percentiles for {}".format(percentile))
        #function for binned_statistic
        pct=percentile[0] #just for a moment, this allows the function definition to work
        for pct in percentile:
            print("... computing percentile {}".format(str(pct)))
            def get_pctl(vec,pctl=pct): #change input pctl variable, XXX:TODO: proper way to do this...?
                return np.percentile(vec,pctl)
            surr_r_bin_pctl[str(pct)] = binned_statistic_2d(surr_HE1,surr_HE2,surr_r,statistic=get_pctl,bins=n_bins_r_cut).statistic.ravel()
    if verbosity > 1:
        print("  After percentile cut ({}): {:.2f}s ".format(percentile, time.time() - st_r), end='\n')

    # the built in functions here are very fast, so using these here rather than deriving bin centeres from the edges
    _surr_HE1_binm_vec = binned_statistic_2d(surr_HE1,surr_HE2,surr_HE1,statistic='mean',bins=n_bins_r_cut).statistic.ravel()
    if verbosity > 1:
        print("  After HE1 mean binning: {:.2f}s ".format(time.time() - st_r), end='\n')
    _surr_HE2_binm_vec = binned_statistic_2d(surr_HE1,surr_HE2,surr_HE2,statistic='mean',bins=n_bins_r_cut).statistic.ravel()
    if verbosity > 1:
        print("  After HE2 mean binning: {:.2f}s ".format(time.time() - st_r), end='\n')
    surr_r_bin_count = binned_statistic_2d(surr_HE1,surr_HE2,surr_r,statistic='count',bins=n_bins_r_cut).statistic.ravel()
    if verbosity > 1:
        print("  After surrogate bin cnt:  {:.2f}s ".format(time.time() - st_r), end='\n')

    #remove NaNs
    if remove_nans:
        if isinstance(percentile,int) or isinstance(percentile,float):
            mask = ~np.isnan(surr_r_bin_pctl)
            surr_r_bin_pctl = surr_r_bin_pctl[mask]
        else:
            mask = ~np.isnan(surr_r_bin_pctl[next(iter(surr_r_bin_pctl))]) #use only the first pctl (key) for computing the nans mask ##TODO: might need to do this for each one :-/
            for key in surr_r_bin_pctl.keys():
                surr_r_bin_pctl[key] = surr_r_bin_pctl[key][mask]
        _surr_HE1_binm_vec = _surr_HE1_binm_vec[mask]
        _surr_HE2_binm_vec = _surr_HE2_binm_vec[mask]
        surr_r_bin_count = surr_r_bin_count[mask]
 
    res_dict = {'surr_HE1_binm_vec': _surr_HE1_binm_vec, 'surr_HE2_binm_vec': _surr_HE2_binm_vec,
                    'surr_r_bin_pctl': surr_r_bin_pctl,'surr_r_bin_count':surr_r_bin_count,'HE1':HE1,'HE2':HE2,'surr_r_pctl_cut':surr_r_pctl_cut}
    return res_dict


@jit(parallel=True,nopython=False)
def parallel_surr_corr(d_sub,r,r1,c,c1,dtype,num_surrogate_reps):
    # _dm1 = np.zeros((r1 - r, c1 - c, num_surrogate_reps))
    _dm1 = np.zeros((d_sub[r:r1].shape[0], d_sub[c:c1].shape[0], num_surrogate_reps)).astype(dtype) #correct for missing voxels when 2d matrix passed

    for surr_idx in range(num_surrogate_reps):

        # grab the original data before computing the surrogates
        chunk1 = np.copy(d_sub[r:r1])
        chunk2 = np.copy(d_sub[c:c1])

        # calculate the surrogates
        for rr in prange(chunk1.shape[0]):
            chunk1[rr, :] = compute_paired_surrogates(chunk1[rr, :], num_surrogates=1)
        if c == r:  # if we are in a diagonal, no need to calculate the surrogates again
            chunk2 = chunk1
        else:
            for rr in prange(chunk2.shape[0]):
                chunk2[rr, :] = compute_paired_surrogates(chunk2[rr, :], num_surrogates=1)
        _dm1[:, :, surr_idx] = np.abs(np.dot(chunk1, chunk2.T).astype(
            dtype))  # compute the data matrices (_dm), using abs because it is the same on both sides (.95, or .975 if two-sided)
    return _dm1


def poly4_2d_predict(xy, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19,
                     b20, b21, b22, c):
    # This should cover all permutations of 4th order polynomial
    x = xy[:, 0]
    y = xy[:, 1]
    return c + b1 * x + b2 * (x ** 2) + b3 * (x ** 3) + b4 * (x ** 4) + b5 * (y) + b6 * (y ** 2) + b7 * (
            y ** 3) + b8 * (y ** 4) + b9 * (x * y) + b10 * ((x ** 2) * (y)) + b11 * ((x ** 3) * (y)) + b12 * (
                   (x ** 4) * (y)) + b13 * ((x) * (y ** 2)) + b14 * ((x) * (y ** 3)) + b15 * (
                   (x) * (y ** 4)) + b16 * ((x ** 2) * (y ** 2)) + b17 * ((x ** 2) * (y ** 3)) + b18 * (
                   (x ** 2) * (y ** 4)) + b19 * ((x ** 3) * (y ** 2)) + b20 * ((x ** 3) * (y ** 3)) + b21 * (
                   (x ** 3) * (y ** 4)) + b22 * ((x ** 4) * (y ** 4))

# @jit(parallel=True) #likely able to go much faster, but you need to cast to float32 arrays
def symmetric_poly4_2d_predict(xy, b1, b2, b3, b4, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19,
                               b20, b21, b22, c):
    # This should cover all permutations of 4th order polynomial
    x = xy[:, 0]
    y = xy[:, 1]
    return c + b1 * x + b2 * (x ** 2) + b3 * (x ** 3) + b4 * (x ** 4) + b1 * (y) + b2 * (y ** 2) + b3 * (
            y ** 3) + b4 * (y ** 4) + b9 * (x * y) + b10 * ((x ** 2) * (y)) + b11 * ((x ** 3) * (y)) + b12 * (
                   (x ** 4) * (y)) + b13 * ((x) * (y ** 2)) + b14 * ((x) * (y ** 3)) + b15 * (
                   (x) * (y ** 4)) + b16 * ((x ** 2) * (y ** 2)) + b17 * ((x ** 2) * (y ** 3)) + b18 * (
                   (x ** 2) * (y ** 4)) + b19 * ((x ** 3) * (y ** 2)) + b20 * ((x ** 3) * (y ** 3)) + b21 * (
                   (x ** 3) * (y ** 4)) + b22 * ((x ** 4) * (y ** 4))

def poly3_2d_predict(xy, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, c):
    # This should cover all permutations of 3rd order polynomial
    x = xy[:, 0]
    y = xy[:, 1]
    return c + b1 * x + b2 * (x ** 2) + b3 * (x ** 3) + b4 * (y) + b5 * (y ** 2) + b6 * (y ** 3) +  b7 * (x * y) + b8 * ((x ** 2) * (y)) + b9 * ((x ** 3) * (y)) + b10 * ((x) * (y ** 2)) + b11 * ((x) * (y ** 3)) + \
           b12 * ((x ** 2) * (y ** 2)) + b13 * ((x ** 2) * (y ** 3)) + b14 * ((x ** 3) * (y ** 2)) + b15 * ((x ** 3) * (y ** 3))

def symmetric_poly3_2d_predict(xy, b1, b2, b3, b7, b8, b9, b10, b11, b12, b13, b14, b15, c):
    # This should cover all permutations of 3rd order polynomial
    x = xy[:, 0]
    y = xy[:, 1]
    return c + b1 * x + b2 * (x ** 2) + b3 * (x ** 3) + b1 * (y) + b2 * (y ** 2) + b3 * (y ** 3) +  b7 * (x * y) + b8 * ((x ** 2) * (y)) + b9 * ((x ** 3) * (y)) + b10 * ((x) * (y ** 2)) + b11 * ((x) * (y ** 3)) + \
           b12 * ((x ** 2) * (y ** 2)) + b13 * ((x ** 2) * (y ** 3)) + b14 * ((x ** 3) * (y ** 2)) + b15 * ((x ** 3) * (y ** 3))

def exp_1d_predict(x,a,b,c):
    """
    Simple exponential. This function is to be able to predict the rcut value at any percentile level (within a single HExHE bin).
    """
    return a * (b)**x + c

def rcut_2d_fit(HE1_vec, HE2_vec, r_cut_vec, fit_type='poly4_symm'):
    """
    
    """
    from scipy.optimize import curve_fit

    A = np.vstack([HE1_vec.ravel(),HE2_vec.ravel()]).T

    if fit_type is "poly3":
        popt, pcov = curve_fit(poly3_2d_predict, A, r_cut_vec)
        fit = 0
    elif fit_type is "poly4":
        popt, pcov = curve_fit(poly4_2d_predict, A, r_cut_vec)
        fit = 0
    elif fit_type is "poly3_symm":
        popt, pcov = curve_fit(symmetric_poly3_2d_predict, A, r_cut_vec)
        fit = 0
    elif fit_type is "poly4_symm":
        popt, pcov = curve_fit(symmetric_poly4_2d_predict, A, r_cut_vec)
        fit = 0
    elif fit_type is "chebyshev3":
        popt, fit = general_symmetric_chebyshev2d_fit(A[:, 0], A[:, 1], r_cut_vec, degree=3)
        pcov = 0
    elif fit_type is "chebyshev4":
        popt, fit = general_symmetric_chebyshev2d_fit(A[:,0],A[:,1],r_cut_vec,degree=4)
        pcov = 0
    elif fit_type is "chebyshev5":
        popt, fit = general_symmetric_chebyshev2d_fit(A[:,0],A[:,1],r_cut_vec,degree=5)
        pcov = 0
    else:
        print('No other fit types currently implemented')
    return {'fit_type':fit_type,'popt':popt,'pcov':pcov,'fit':fit}


def rcut_2d_predict(HE1_vec, HE2_vec, popt, fit_type='chebyshev'):
    A = np.vstack([HE1_vec.ravel(),HE2_vec.ravel()]).T #use ravel to push into assumed vector form
    if fit_type is 'poly3':
        pred_rcut = poly3_2d_predict(A, *popt)
    elif fit_type is 'poly4':
        pred_rcut = poly4_2d_predict(A, *popt)
    elif fit_type is 'poly3_symm':
        pred_rcut = symmetric_poly3_2d_predict(A, *popt)
    elif fit_type is 'poly4_symm':
        pred_rcut = symmetric_poly4_2d_predict(A, *popt)

    elif fit_type is "chebyshev3":
        pred_rcut = general_symmetric_chebyshev2d_predict(A[:,0],A[:,1],popt,degree=3)
    elif fit_type is "chebyshev4":
        pred_rcut = general_symmetric_chebyshev2d_predict(A[:,0],A[:,1],popt,degree=4)
    elif fit_type is "chebyshev5":
        pred_rcut = general_symmetric_chebyshev2d_predict(A[:,0],A[:,1],popt,degree=5)
    return pred_rcut


def general_symmetric_chebyshev2d_fit(x_val, y_val, z_val, degree=4, min_val=None, max_val=None):
    # matrix of (space x coeffs)
    coeffs = general_symmetric_chebyshev2d_coeffs(x_val, y_val, degree, min_val, max_val)

    # linear system solving
    model = np.linalg.lstsq(coeffs, z_val, rcond=None)[0]

    # compute error
    fit = np.dot(coeffs, model)
    error = np.sqrt(np.mean((fit - z_val) * (fit - z_val)))
    print("Chebyshev fit error: " + str(error))

    return model, fit


def general_symmetric_chebyshev2d_predict(x_val, y_val, model, degree=4, min_val=None, max_val=None):
    # matrix of (space x coeffs)
    coeffs = general_symmetric_chebyshev2d_coeffs(x_val, y_val, degree, min_val, max_val)

    # linear system solving
    fit = np.dot(coeffs, model)

    return fit


def general_symmetric_chebyshev2d_coeffs(x_val, y_val, degree=4, min_val=None, max_val=None):
    dims = len(x_val)
    if len(y_val) != dims:
        print("x and y lists should have the same dimensions")
        return None

    if degree is 1:
        dims = (dims, 2)
    elif degree is 2:
        dims = (dims, 4)
    elif degree is 3:
        dims = (dims, 6)
    elif degree is 4:
        dims = (dims, 9)
    elif degree is 5:
        dims = (dims,12)

    xymin = np.min([np.min(x_val), np.min(y_val)])
    xymax = np.max([np.max(x_val), np.max(y_val)])

    if min_val is not None:
        if min_val < xymin:
            xymin = min_val
        else:
            print("minimum bound higher than some values, incorrect for Chebyshev approximation")
            return None

    if max_val is not None:
        if max_val > xymax:
            xymax = max_val
        else:
            print("maximum bound lower than some values, incorrect for Chebyshev approximation")
            return None

    xym = (xymax - xymin) / 2.0

    coeffs = np.zeros(dims)

    for idx in range(dims[0]):
        x = x_val[idx]
        y = y_val[idx]

        # ratio values in [-1,1]
        xr = (x - xym) / xym
        yr = (y - xym) / xym

        # coefficients
        if degree >= 1:
            coeffs[idx, 0] = 1.0
            coeffs[idx, 1] = xr + yr
        if degree >= 2:
            coeffs[idx, 2] = 2.0 * xr * xr - 1.0 + 2.0 * yr * yr - 1.0
            coeffs[idx, 3] = xr * yr
        if degree >= 3:
            coeffs[idx, 4] = 4.0 * xr * xr * xr - 3.0 * xr + 4.0 * yr * yr * yr - 3.0 * yr
            coeffs[idx, 5] = (2.0 * xr * xr - 1.0) * yr + (2.0 * yr * yr - 1.0) * xr
        if degree >= 4:
            coeffs[
                idx, 6] = 8.0 * xr * xr * xr * xr - 8.0 * xr * xr + 1.0 + 8.0 * yr * yr * yr * yr - 8.0 * yr * yr + 1.0
            coeffs[idx, 7] = (4.0 * xr * xr * xr - 3.0 * xr) * yr + (4.0 * yr * yr * yr - 3.0 * yr) * xr
            coeffs[idx, 8] = (2.0 * xr * xr - 1.0) * (2.0 * yr * yr - 1.0)
        if degree>=5:
            coeffs[idx,9] = 16.0*xr*xr*xr*xr*xr - 20.0*xr*xr*xr + 5.0*xr \
                            + 16.0*yr*yr*yr*yr*yr - 20.0*yr*yr*yr + 5.0*yr
            coeffs[idx,10] = (8.0*xr*xr*xr*xr - 8.0*xr*xr + 1.0)*yr \
                            + (8.0*yr*yr*yr*yr - 8.0*yr*yr + 1.0)*xr
            coeffs[idx,11] = (4.0*xr*xr*xr - 3.0*xr)*(2.0*yr*yr - 1.0) \
                            + (4.0*yr*yr*yr - 3.0*yr)*(2.0*xr*xr - 1.0)

    return coeffs
