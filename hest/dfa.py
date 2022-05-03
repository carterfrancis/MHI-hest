# import nibabel as nb
import numpy as np
import time
import scipy
import scipy.signal
import h5py
from numba import jit, prange

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



def dfa_blocked(img, mask=None, detrend_type='linear', min_samples=5, max_samples=50, num_window_steps=5,
                manual_sample_nums_list = None, flip_window_when_uneven=False,
                out_dir=None, by_voxel_group=150, global_mean_signal_regression=False,
                verbosity=0, output_window_data=False, data_type='float32',
                clean_timecourse_data=False, apply_correction=None):
    """
    Perform detrended fluctuation analysis (dfa) on all data within the rsfMRI image (img). Returns Hurst Exponent (HE)
    and the R2 of the log10-log10 fit that calculated it. If output_window_data==True, variance computed for each window set
    will also be output with a text file denoting the number of samples per window step (useful for offline computation
    as necessary).

    :param img: 4D nibabel image or full filename, 2D timecourse of elementsXsamples can also be used
    :param mask: mask (nibabel image or full filename) of elements where the HE should be calculated (==1) If mask is not provided,
        non-zero values from the first 3d volume will be used
    :param detrend_type: detrending type for each window - currently only supports 'linear'
    :param min_samples: minimum number of samples per window (usually set to >= 5)
    :param max_samples: maximum number of samples per window (should be include min_samples^10 or greater)
    :param num_window_steps: number of windowing steps used for calculation, samples per window will be increasing in
        log space, based on the floor of np.geomspace(min_samples, max_samples, num=num_window_steps)
    :param manual_sample_nums_list: defines a list of sample sizes for each window, overrides min_samples, max_samples,
        and num_window_steps
    :param flip_window_when_uneven: implements the window-flip procedure from https://arxiv.org/pdf/cond-mat/0102214.pdf
        to solve the issue of remaining samples after windowing. This increases computation time (up to appx 4x longer)
        but also generally results in slightly higher R2 values when computing HE. When False, remaining samples are
        distributed between windows approximately evenly and mean sample size per window (float) is used as the
        regressor for the log10-log10 fit.
    :param out_dir: output directory, if set, for results
    :param by_voxel_group: number of voxels**3 to work with in each block. Use to save memory when working with large files
    :param global_mean_signal_regression: perform global mean signal regression (regress out the timeseries consisting
        of the mean of all elements in each timepoint
    :param verbosity: controls the level of text output printed to stdount {-1,0,2}, with -1 supressing all output
    :param output_window_data: also save/return window data, including std in each window and text file with the number
        of samples per window for each window step that was used in the HE calculation
    :param data_type: data type to work with the data as, and output as
    :param clean_timecourse_data: cleans element timecourses that include nan, +/- inf and very large/small numbers
    :param apply_correction: if not None, corrects dfa based on the variance of randomised time series from a single
            randomly selected element
            based on Kantelhardt et al., 2001 (https://arxiv.org/abs/cond-mat/0102214)
    :return:
        If output_window_data is False: dictionary containing nibabel images for HE and R2 fit
            {'HE': img_HE, 'HE_R2': img_R2}
        If output_window_data is True: dictionary containing HE, R2, std per window (windows), number of samples per
            window (windows_num_samples), and standard deviation of the raw/cleaned timecourse (timecourse_std)
            {'HE': img_HE, 'HE_R2': img_R2, 'windows':img_dfa_wins,
                    'windows_num_samples':actual_samples_per_window, 'timecourse_std':img_total_std}
        If img is a 2D matrix, returns all results as matrices and does not save any files
    """

    st_orig = time.time()

    img_input_output = True

    if by_voxel_group is None:  # if none provided, we just make the voxel group arbitrarily large
        by_voxel_group = int(10e10)
    if isinstance(img, str):
        fname = img
        img = nb.load(img)
    elif isinstance(img,np.ndarray):
        img_input_output = False
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

    ## set up the computations for the blocked analyses ##
    # determine where data is in the x,y,z directions via mask
    all_nonzero = np.array(np.where(z_mask))
    min_max = np.vstack((np.min(all_nonzero, axis=1), np.max(all_nonzero, axis=1)))

    # break data into blocks based on the size and the by_voxel_group number
    # fancy indexing does not work, at all, block indexing does, but annoying
    dim_diff = np.diff(min_max, axis=0)[0] + 1  # XXX TODO, CHECK
    largest_dim = np.argmax(dim_diff)  # this is the dim that we will split on
    if verbosity >0:
        print("Dimension difference for block computation (split on largest): {}".format(dim_diff))

    x_idxs = min_max[:, 0]
    y_idxs = min_max[:, 1]
    z_idxs = min_max[:, 2]
    blocks = np.array_split(np.arange(dim_diff[largest_dim]),
                            np.ceil(dim_diff[largest_dim] / float(by_voxel_group)))
    # set our output arrays that we will fill (weeee)
    HE_out = np.zeros(z_mask.shape).astype(float)  # only want 3d img
    R2_out = np.zeros(z_mask.shape).astype(float)

    # 2) create windows
    if img_input_output:
        t_pts = img.get_shape()[3] #pull the size of the last dim from the 4d file
    else:
        t_pts = img.shape[1] #get the size of the last dim in 2d input file

    if manual_sample_nums_list is not None:
        if verbosity > -1:
            print('Sample sizes per window were set manually')
        manual_sample_nums_list = np.sort(manual_sample_nums_list) #samples go from few to many per window
        window_nums = t_pts/manual_sample_nums_list
        num_window_steps = len(window_nums)
    else:
        if verbosity > -1:
            print('Sample sizes per window were calculated based on min_samples, max_samples, and num_window_steps')
        # we generate given the num_window_steps
        # we go from smaller num samples up to larger
        appx_samples_per_window = np.floor(np.geomspace(min_samples, max_samples, num=num_window_steps))
        window_nums = np.round(float(t_pts) / appx_samples_per_window.astype(float)).astype(int)
        _, idx_unique = np.unique(window_nums, return_index=True)

        #if too many steps were provided, we remove the ones that are identical
        removed_win_cnt = 0
        while len(idx_unique) < len(window_nums):
            removed_win_cnt +=1
            if (verbosity > -1) and (removed_win_cnt==1):
                print("Redundant window(s) detected:")
            if verbosity > -1:
                print("\t--> removing redundant window:\t{:2d}".format(removed_win_cnt))
            num_window_steps -= 1
            appx_samples_per_window = np.floor(np.geomspace(min_samples, max_samples, num=num_window_steps))
            window_nums = np.round(float(t_pts) / appx_samples_per_window.astype(float)).astype(int)
            _, idx_unique = np.unique(window_nums, return_index=True)

    if verbosity > -1:
        print("Number of window steps used in this dataset: {}".format(num_window_steps))
    if output_window_data:
        dfa_windows_out = np.zeros(z_mask.shape+tuple([num_window_steps]))
        total_std_out = np.zeros(z_mask.shape)

    actual_samples_per_window = float(t_pts) / window_nums.astype(float)
    if flip_window_when_uneven: #we alter the way that we perform the sampling, require ints for splitting
        actual_samples_per_window = np.floor(actual_samples_per_window).astype(int)

    if verbosity > -1:
        print("Iterating over {} 3d blocks of data, with a total of {} sample timepoints".format(len(blocks) ** 3,t_pts))
        print("Performing dfa with {} sets of windows:".format(num_window_steps))
        print("======================================================================================================")
        print("\tInput data dimensions:\t\t\t{}".format(img.shape))
        print("\tNumber of windows used for HE calc:\t{}".format(window_nums))
        print("\tNumber of samples in each window:\t" + str(
            ["{0:0.2f}".format(i) for i in actual_samples_per_window]).replace("'", ""))
        print("======================================================================================================")
        print("")

    if global_mean_signal_regression:
        print("Global mean signal regression will be performed")
        global_mean_ts = compute_global_mean_signal(img,mask=z_mask)

    blk_counter = 0
    for idx, block1 in enumerate(np.copy(blocks)):  # iteration in x,y,z to get all combinations of locations
        for idx2, block2 in enumerate(np.copy(blocks)):
            for idx3, block3 in enumerate(np.copy(blocks)):
                st_block = time.time()
                blk_counter += 1

                if verbosity > 0:
                    print("\n---> Block counter: {}".format(blk_counter))
                    print('Block index: {},{},{}'.format(idx, idx2, idx3))

                x_coords = block1 + x_idxs[0]
                y_coords = block2 + y_idxs[0]
                z_coords = block3 + z_idxs[0]

                if verbosity > 1:
                    print("  x,y,z range: {}-{}, {}-{}, {}-{}".format(x_coords[0], x_coords[-1] + 1, y_coords[0],
                                                                      y_coords[-1] + 1, z_coords[0], z_coords[-1] + 1))
                # pull the data from the locations in the mask to determine if we actually have data here
                # edge blocks may not have data, because I didn't specify coordinates in an exact way
                if img_input_output:
                    m_sub = z_mask[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1, z_coords[0]:z_coords[-1] + 1]
                else:
                    m_sub = np.squeeze(z_mask[:,0])
                if np.sum(m_sub) == 0:
                    if verbosity > 0:
                        print("  No data here, skipping to next block (this is normal)")
                else:
                    # grab between the first and last of the block, extract all values in 4th dim
                    if img_input_output:
                        d_sub = np.array(img.dataobj[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1,
                                         z_coords[0]:z_coords[-1] + 1, :]).astype(data_type)
                    else: #in this case, we are using a 2D array
                        d_sub = np.squeeze(img)

                    ## calculate Hurst exponent (HE)
                    # 0) clean the data of any nan, inf, too large nums by masking the data
                    if clean_timecourse_data:
                        d_sub = _clean_timecourse(d_sub[m_sub, :])
                    else:
                        d_sub = d_sub[m_sub,:]

                    if global_mean_signal_regression: #this should do global mean signal regression, though not tested against other implementations
                        _model, _resid, _rank, _s = np.linalg.lstsq(np.vstack([np.ones_like(global_mean_ts),global_mean_ts]).T, d_sub.T, rcond=None)  # simple linear fit
                        _cs, _ms = _model
                        _pred = (global_mean_ts[:,None] * _ms + _cs).T
                        d_sub = d_sub - _pred

                    # 1) cumulative sum after removing mean of each timeseries
                    data_masked_cumsum = np.cumsum(d_sub - np.mean(d_sub, axis=1)[..., np.newaxis], axis=1)

                    # a) IF we want to correct the DFA alpha estimate, we only need to do this once (according to the paper)
                    if (blk_counter == 1) and (apply_correction is not None):
                        window_nums_shuff = np.copy(window_nums)
                        if apply_correction < 1000:
                            apply_correction = 1000
                        print('Using {} randomised samples of a single element to compute alpha=0.5 variance correction factor'.format(apply_correction))
                        data_masked_cumsum_shuff = np.zeros((apply_correction, t_pts))
                        ts_idx = np.random.choice(np.arange(d_sub.shape[0]))
                        for rand_idx in np.arange(apply_correction):
                            data_masked_cumsum_shuff[rand_idx, :] = data_masked_cumsum[ts_idx, np.random.permutation(t_pts)]

                        # b) create windows of data for shuffled data
                        # also need to compute the variance for a "standard" size with enough good data
                        # samples_per_window ~ t_pts/20 AND samples_per_window >= 50

                        xtra_window_tpts_per = np.floor(t_pts / 20.).astype(int)
                        if xtra_window_tpts_per < 50.:
                            print('>>>>>>>> Warning: you do not have enough time points in this data to accurately correct it. <<<<<<<<')
                            print('>>>>>>>> attempting correction with an intermediate number of samples per window: {}      <<<<<<<<'.format(xtra_window_tpts_per))
                        else:
                            print('You seem to have enough timepoints to accuractely correct alpha for this data, correcting with variance estimated from {} samples per window'.format(xtra_window_tpts_per))
                        window_nums_shuff = np.hstack([window_nums_shuff,(t_pts/xtra_window_tpts_per).astype(int)])
                        print(window_nums_shuff )
                        N_var_mat_shuff = np.zeros((data_masked_cumsum_shuff.shape[0], len(window_nums_shuff)))
                        for idx_win, window_num in enumerate(window_nums_shuff):

                            if flip_window_when_uneven:
                                # if the timecourse is not evenly divisible by the sample size, we append windows from the reversed timecourse
                                # in order to recapture lost samples
                                extra_samples = np.remainder(t_pts, window_num)
                                if extra_samples > 0:
                                    wins = np.split(data_masked_cumsum_shuff[:, :-extra_samples], window_num, axis=1) + \
                                           np.split(data_masked_cumsum_shuff[:, ::-1][:, :-extra_samples], window_num, axis=1)
                                else:
                                    wins = np.split(data_masked_cumsum_shuff, window_num, axis=1)
                            else:  # or we distribute the extra samples throughout
                                wins = np.array_split(data_masked_cumsum_shuff, window_num, axis=1)

                            var_shuff = np.zeros(
                                (data_masked_cumsum_shuff.shape[0], len(wins)))  # to hold variance of each voxelXwindow

                            # for each window
                            for idx_win2, win in enumerate(wins):
                                # 3) detrend data timecourse(s)
                                # 4) calculate the standard deviation
                                # var[:, idx_win2] = np.std(scipy.signal.detrend(win, type=detrend_type), axis=1)
                                var_shuff[:, idx_win2] = (np.mean(scipy.signal.detrend(win, type='linear') ** 2,
                                                            axis=1)) ** 0.5  # same results, slightly greater speed
                            # 5) average over the windows in this window_num and keep track (mean variance with N windows)
                            N_var_mat_shuff[:, idx_win] = np.nanmean(var_shuff, axis=1)
                        N_var_mat_shuff = N_var_mat_shuff[:,0:-1]/N_var_mat_shuff[:,-1][...,np.newaxis]

                    # 2) create windows of data
                    N_var_mat = np.zeros((data_masked_cumsum.shape[0], len(window_nums)))
                    for idx_win, window_num in enumerate(window_nums):
                        if verbosity > 1:
                            print('idx_win: {}'.format(idx_win)),
                        st = time.time()

                        if flip_window_when_uneven:
                            # if the timecourse is not evenly divisible by the sample size, we append windows from the reversed timecourse
                            # in order to recapture lost samples
                            extra_samples = np.remainder(t_pts, window_num)
                            if extra_samples > 0:
                               wins = np.split(data_masked_cumsum[:, :-extra_samples], window_num, axis=1) + \
                                       np.split(data_masked_cumsum[:, ::-1][:, :-extra_samples], window_num, axis=1)
                            else:
                                wins = np.split(data_masked_cumsum, window_num, axis=1)
                        else: #or we distribute the extra samples throughout
                            wins = np.array_split(data_masked_cumsum, window_num, axis=1)

                        var = np.zeros((data_masked_cumsum.shape[0], len(wins))) #to hold variance of each voxelXwindow

                        # for each window
                        for idx_win2, win in enumerate(wins):
                            if verbosity > 1:
                                print('idx_win2: {}'.format(idx_win2)),
                            # 3) detrend data timecourse(s)
                            # 4) calculate the standard deviation
                            #var[:, idx_win2] = np.std(scipy.signal.detrend(win, type=detrend_type), axis=1)
                            var[:, idx_win2] = (np.mean(scipy.signal.detrend(win, type='linear') ** 2, axis=1))**0.5 #same results, slightly greater speed?
                        # 5) average over the windows in this window_num and keep track (mean variance with N windows)
                        N_var_mat[:, idx_win] = np.nanmean(var, axis=1)
                        et = time.time()
                        if verbosity > 1:
                            print(
                            '  Window idx {} with {} windows, appx {} samples per window'.format(idx_win, window_num, (
                            float(t_pts) / float(window_num))))
                            print("  elapsed: {0:.2f}s".format(et - st))

                    if verbosity > -1:
                        print("  Time for window variance calculations in block {2}: {0:.2f}s for {1} windows".format(
                        time.time() - st_block, num_window_steps, blk_counter))

                    if apply_correction:
                        N_var_mat /= N_var_mat_shuff.mean(axis=0)

                    # 6) linear regression of sample numbers (per window) and mean variance to calc HE
                    # slope of the line == Hurst xp and R2 values calculated to confirm that our estimates are good
                    x = actual_samples_per_window

                    N_var_mat_mask = (N_var_mat==0)
                    if np.sum(N_var_mat_mask) > 0:
                        if verbosity > -1:
                            print('{} zeros detected in your dataset (likely due to incorrect masking), these voxels will be masked from the analysis'.format(np.sum(N_var_mat_mask)))
                        N_var_mat_mask[np.unique(np.array(np.where(N_var_mat_mask))[0,:]),:] = True #mask out the entire row of data, so that we don't get artefactual values
                        y = np.ma.log10(np.ma.masked_array(N_var_mat, N_var_mat_mask).T)
                    else:
                        y = np.log10(N_var_mat.T)
                    X = np.vstack([np.ones(len(x)), np.log10(x)]).T
                    model, resid, rank, s = np.linalg.lstsq(X, y, rcond=None)  # simple linear fit

                    # checked R2 calc, much faster than with SSresid and SStot
                    # also used masked computations here, in case we are missing values
                    # https://stackoverflow.com/questions/3054191/converting-np-lstsq-residual-value-to-r2
                    t_R2 = 1 - np.ma.divide(resid, np.ma.multiply(y.shape[0], np.ma.std(y,
                                                                               axis=0) ** 2))
                    cs, t_HE = model  # X setup so that slopes are the 2nd val

                    if img_input_output:
                        # return HE_out, t_HE indexed by both coords defining the sub-volume and m_sub defining the regions with data
                        HE_out[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1, z_coords[0]:z_coords[-1] + 1][
                            m_sub] = t_HE
                        R2_out[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1, z_coords[0]:z_coords[-1] + 1][
                            m_sub] = t_R2
                        if output_window_data:
                            dfa_windows_out[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1, z_coords[0]:z_coords[-1] + 1,:][
                            m_sub] = N_var_mat
                            total_std_out[x_coords[0]:x_coords[-1] + 1, y_coords[0]:y_coords[-1] + 1, z_coords[0]:z_coords[-1] + 1][
                            m_sub] = np.nanstd(d_sub.astype(float),axis=-1) #cast to float for this computation in case lower precision was specified
    if img_input_output:
        # Turn back into an img
        head['cal_min'] = np.min(HE_out)
        head['cal_max'] = np.max(HE_out)
        img_HE = nb.Nifti1Image(HE_out, aff, header=head)

        head['cal_min'] = np.min(R2_out)
        head['cal_max'] = np.max(R2_out)
        img_R2 = nb.Nifti1Image(R2_out, aff, header=head)

        if output_window_data:
            head['cal_min'] = np.min(dfa_windows_out)
            head['cal_max'] = np.max(dfa_windows_out)
            img_dfa_wins = nb.Nifti1Image(dfa_windows_out, aff, header=head)

            head['cal_min'] = np.min(total_std_out)
            head['cal_max'] = np.max(total_std_out)
            img_total_std = nb.Nifti1Image(total_std_out, aff, header=head)

        if out_dir is not None:
            from os.path import join
            if fname is not None:
                fname_head = fname.split('/')[-1].split('.')[0]
            else:
                fname_head = "Hurst_Exponent"
            HE_out_name = join(out_dir, fname_head + "_" + detrend_type + "_" + str(num_window_steps) + "win" + "_dfa_HE.nii.gz")
            R2_out_name = join(out_dir, fname_head + "_" + detrend_type + "_" + str(num_window_steps) + "win" +  "_dfa_HE_R2.nii.gz")
            if output_window_data:
                wins_out_name = join(out_dir, fname_head + "_" + detrend_type + "_" + str(num_window_steps) + "win" +  "_dfa_windows.nii.gz")
                total_std_out_name = join(out_dir, fname_head + "_" + detrend_type + "_" + str(num_window_steps) + "win" +  "_dfa_timecourse_std.nii.gz")

            try:
                img_R2.to_filename(R2_out_name)
                img_HE.to_filename(HE_out_name)
                if output_window_data:
                    img_dfa_wins.to_filename(wins_out_name)
                    img_total_std.to_filename(total_std_out_name)
                    np.savetxt(wins_out_name.split('.nii')[0]+".txt",np.vstack((window_nums.astype(float),actual_samples_per_window)),fmt='%.5f')
                    if verbosity > -1:
                        print("Image files written to:\n{0}\n{1}\n{2}\n{3}\n".format(HE_out_name, R2_out_name, wins_out_name,total_std_out_name))
                else:
                    if verbosity > -1:
                        print("Image files written to:\n{0}\n{1}\n".format(HE_out_name, R2_out_name))
            except:
                if verbosity > -1:
                    print("Image files were not written to disk properly")
        if verbosity > -1:
            print("======================================================================================================")
            print("\tMean non-zero HE: \t\t\t{:.3f} ".format(np.mean(HE_out[HE_out>0])))
            print("\tMean non-zero R2 fit: \t\t\t{:.3f} ".format(np.mean(R2_out[R2_out > 0])))
            print("======================================================================================================")
            print("Total time for HE calc: {0:.2f}s".format(time.time() - st_orig))
            print("")
        if out_dir is not None:
            if output_window_data:
                return {'HE': img_HE, 'HE_R2': img_R2, 'var_per_window':img_dfa_wins, 'num_windows':window_nums,
                        'samples_per_window':actual_samples_per_window, 'timecourse_std':img_total_std}
            else:
                return {'HE': img_HE, 'HE_R2': img_R2}
        else:
            if output_window_data:
                return {'HE': img_HE, 'HE_R2': img_R2, 'var_per_window':img_dfa_wins, 'num_windows':window_nums,
                        'samples_per_window':actual_samples_per_window, 'timecourse_std':img_total_std}
            return {'HE':img_HE, 'HE_R2':img_R2}
    else:
        if verbosity > -1:
            print("======================================================================================================")
            print("\tMean non-zero HE: \t\t\t{:.3f} ".format(np.mean(t_HE[t_HE>0])))
            print("\tMean non-zero R2 fit: \t\t\t{:.3f} ".format(np.mean(t_R2[t_R2> 0])))
            print("======================================================================================================")
            print("Total time for HE calc: {0:.2f}s".format(time.time() - st_orig))
            print("")
        return {'HE':np.array(t_HE),'HE_R2':np.array(t_R2),'var_per_window':N_var_mat, 'num_windows':window_nums,
                'timecourse_std':np.nanstd(d_sub.astype(float),axis=-1),'samples_per_window':actual_samples_per_window}

