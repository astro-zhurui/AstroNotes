import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.table import Table
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Zout():

    def __init__(self, param_file=None, output_dir=None, output_file=None, cache_file=None, read_bin=1):
        """
        Can read eazy output files and plot figures.
        Reading eazy parameter file may not work when data analysis and estimating photo-z
        are not done in the same environment. So output file path need to be set manually.
        # Parameters\n
        param_file: str | bool
          Use EAZY parameter file directly
        output_dir: str | bool
          output directory
        output_file: str | bool
          output file name, NO file extension
        cache_file: str | bool
          cache file name, file extension is needed
        read_bin: int | bool | NoneType
          will skip the process of read binary output file if set to 0 or None of False

        # Example\n
          Either of the following usage is ok
        >>> zout = Zout(param_file='xxx/xxx/zphot.param')

         or

        >>> zout = Zout(output_dir='xxx/xxx/', output_file='xxx', cache_file='xxx.tempfilt')

          after loading the output file the following functions can be used

        >>> zout.plot_all()  
        
          plot z_phot-z_spec, sed, pdf in the same figure, use mouse to select from the z_phot-z_spec figure, sed and pdf of the choosen galaxy will be shown
        
        >>> zout.zphot_zspec()

          plot z_phot-z_spec

        >>> zout.show_fitting()

          plot sed and pdf

        >>> zout.show_sed_fitting()

          plot sed only

        >>> zout.show_pdf()

          plot pdf only
        """
        # load parameters
        if param_file:
            param_dict = read_eazy_param(param_file)
            read_bin = param_dict['BINARY_OUTPUT']
            output_dir = param_dict['OUTPUT_DIRECTORY']
            output_file = param_dict['MAIN_OUTPUT_FILE']
            cache_file = param_dict['CACHE_FILE']
        # load binary output files
        if read_bin:
            self.tempfilt, self.coeffs, self.temp_sed, self.pz, self.photoz_info \
                = read_eazy_binary(output_dir, output_file, cache_file)
        zout_path = os.path.join(output_dir, output_file+'.zout')
        zout = Table.read(zout_path, format='ascii.commented_header')
        zout_col = list(zout.columns)
        self.z_best = zout['z_peak']
        self.z_spec = zout['z_spec']
        self.z_u68 = zout['u68']
        self.z_l68 = zout['l68']
        self.id = zout['id']
        self.qz = zout['q_z']
        # diffenent types of templates combo have different output file
        # need to be classified before load template information
        if 'temp_p' in zout_col:        # single template fitting with prior
            self.temp = zout['temp_p']
            self.chi2 = zout['chi_p']
            self.combo = 1
        elif 'temp_1' in zout_col:      # single template fitting without prior
            self.temp = zout['temp_1']
            self.chi2 = zout['chi_1']
            self.combo = 1
        elif 'temp_pa' in zout_col:     # double template fitting with prior
            self.temp = [zout['temp_pa'], zout['temp_pb']]
            self.chi2 = zout['chi_p']
            self.combo = 2
        elif 'temp2a' in zout_col:      # double template fitting without prior
            self.temp = [zout['temp2a'], zout['temp_2b']]
            self.chi2 = zout['chi_2']
            self.combo = 2
        elif 'chi_p' in zout_col:       # full template set fitting with prior
            self.temp = np.zeros(len(zout), dtype=int)
            self.chi2 = zout['chi_p']
            self.combo = 99
        else:                           # full template set fitting without prior
            self.temp = np.zeros(len(zout), dtype=int)
            self.chi2 = zout['chi_a']
            self.combo = 99

    def plot_all(self, idx=None, zmax=6, errorbar=False, lrange=[2000, 11000], log_wavelen=False, individual_templates=False):
        """
        Plot z_phot-z_spec, sed, pdf in the same figure, use mouse to select from
        the z_phot-z_spec figure, sed and pdf of the choosen galaxy will be shown
        # Parameters\n
        idx: array_like | NoneType
          for zphot-zspec subfigure, plot the result of galaxies in the given index
        zmax: float
          for zphot-zspec subfigure, max redshift of z_best and z_spec axis, only influence the display area
        lrange: array_like
          for sed fitting subfigure, decide the lower and upper limit of wavelength display range
        log_wavelen: bool
          for sed fitting subfigure, display the wavelength axis in log scale
        individual_templates: bool
          for sed fitting subfigure, display all the SED in sed template sets
        """
        self.lrange = lrange
        self.log_wavelen = log_wavelen
        self.zmax = zmax
        self.individual_temp = individual_templates
        # set subfig location
        self.fig = plt.figure(figsize=(8,6))
        ax_zphot_zspec = plt.axes([0.08,0.42,0.4,0.53])
        self.ax_sed = plt.axes([0.58,0.55,0.4,0.4])
        self.ax_pdf = plt.axes([0.58,0.08,0.4,0.35])
        # plot z_phot-z_spec
        ax_zphot_zspec = self.zphot_zspec(idx=idx, ax=ax_zphot_zspec, show_text=False, 
                                                errorbar=errorbar, zmax=zmax)
        # foreground and background settings for the choosen galaxy
        self.selected_fg, = ax_zphot_zspec.plot(self.z_spec[0], self.z_best[0], 'o', ms=2, 
                                            alpha=.8, color='k', visible=False, zorder=99)
        self.selected_bg, = ax_zphot_zspec.plot(self.z_spec[0], self.z_best[0], 'o', ms=8, 
                                            alpha=.8, color='C3', visible=False, zorder=98)
        # information for photo-z result
        dz, f_out, sigma = zout_analysis(self.z_spec[idx], self.z_best[idx])
        photoz_info_text = 'N:\n$f_{out}$:\n$\sigma_{nMAD}$:'
        photoz_info_data = '{:>7d}\n{:>6.2f}%\n{:>7.3f}'.format(len(dz), f_out, sigma)
        ax_zphot_zspec.text(0, -0.25, photoz_info_text, transform=ax_zphot_zspec.transAxes, 
                            va='top', ha='left', linespacing=1.38)
        ax_zphot_zspec.text(.3, -0.25, photoz_info_data, transform=ax_zphot_zspec.transAxes, 
                            va='top', ha='right', linespacing=1.5)
        # information for selected galaxy
        selected_info_text = 'id:\nidx:\n$z_{spec}:$\n$z_{best}:$\n$q_z:$'
        selected_info_data = 'None\nNone\nNone\nNone\nNone'
        ax_zphot_zspec.text(0.45, -0.25, selected_info_text, transform=ax_zphot_zspec.transAxes, 
                            va='top', ha='left', linespacing=1.38)
        self.selected_info = ax_zphot_zspec.text(0.8, -0.25, selected_info_data, 
                                                 transform=ax_zphot_zspec.transAxes, 
                                                 va='top', ha='right', linespacing=1.5)
        # information for SED fitting
        fitting_info_text = 'temp:\n$\chi^2$:'
        fitting_info_data = 'None\nNone'
        ax_zphot_zspec.text(0, -0.55, fitting_info_text, transform=ax_zphot_zspec.transAxes, 
                            va='top', ha='left', linespacing=1.1)
        self.fitting_info = ax_zphot_zspec.text(0.18, -0.55, fitting_info_data, 
                                                 transform=ax_zphot_zspec.transAxes, 
                                                 va='top', ha='left', linespacing=1.5)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def zphot_zspec(self, idx=None, zmax=6, plot_dz=False, errorbar=False, show_text=True, ax=False):
        """
        plot z_phot-z_spec figure\n
        # parameters
        plot_dz:
          plot dz-z_spec at the bottom of figure
        errorbar:
          show error bar for each source, derived from 'u68' and 'l68' columns in the zout file
        show_text:
          will print number of sources, fraction of outliers and scatter in the lower right corner of the figure       
        """
        ttlo = False # plt.tight_layout()
        if not ax:
            ttlo = True
            if plot_dz:
                plt.figure(figsize=(5*1.2,6*1.2))
            else:
                plt.figure(figsize=(5*1.2,5*1.2))
            ax = plt.subplot(111)
        self.idx = idx
        if type(idx) == bool or idx is None:
            self.idx = np.arange(len(self.z_best))
        # show errorbar        
        if errorbar:
            u_err = self.z_u68[self.idx] - self.z_best[self.idx]
            l_err = self.z_best[self.idx] - self.z_l68[self.idx]
            ax.errorbar(self.z_spec[self.idx], self.z_best[self.idx], yerr=[l_err, u_err], 
                        fmt='o', ms=1, color='k', elinewidth=.25, alpha=.8)   
        # make points selectable
        ax.scatter(self.z_spec[self.idx], self.z_best[self.idx], 
                    s=2, alpha=0.8, c='k', picker=True, pickradius=5)
        ax.plot([0,zmax], [0,zmax], c='k', lw=1)
        ax.plot([0, zmax], [-.15, .85*zmax-.15], linestyle='--', color='C1', linewidth=1)
        ax.plot([0, zmax], [.15, 1.15*zmax+.15], linestyle='--', color='C1', linewidth=1)
        dz, f_out, sigma = zout_analysis(self.z_spec[self.idx], self.z_best[self.idx])
        if show_text:
            # calculate fraction of outliers and sigma
            ax.text(0.95, 0.15, 'N: %d'%len(dz), transform=ax.transAxes, ha='right', size=10)
            ax.text(0.95, 0.10, '$f_{out}:$%.2f%%'%f_out, transform=ax.transAxes, ha='right', size=10)
            ax.text(0.95, 0.05, '$\sigma_{nMAD}:%.3f $'%sigma, transform=ax.transAxes, ha='right', size=10)
        # set x axis and y axis
        ax.set_xlabel('$z_{spec}$')
        ax.set_xlim(0, zmax)
        ax.set_ylabel('$z_{best}$')
        ax.set_ylim(0, zmax)
        # plot z_spec-dz diagram
        if plot_dz:
            ax.xaxis.set_tick_params(labelbottom=False)
            divider = make_axes_locatable(ax)
            ax_bottom = divider.append_axes('bottom', size=1, pad=.1, sharex=ax)
            ax_bottom.scatter(self.z_spec[self.idx], dz, s=2, c='k')
            ax_bottom.plot([-1,zmax+1,zmax+1,-1], [-.15,-.15,.15,.15], '--', c='C1', lw=1)
            ax_bottom.plot([-1,zmax+1], [0,0], c='C3', lw=1)
            # axis settings
            ax_bottom.set_ylim(-.5, .5)
            ax_bottom.set_yticks([-.3, 0, .3])
            ax_bottom.set_xlabel(r'$z_\mathrm{spec}$')
            ax_bottom.set_ylabel('dz')
        
        if ttlo:
            plt.tight_layout()

        return ax

    def show_fitting(self, idx, lrange=[2000, 11000], log_wavelen=False, individual_templates=False, id_is_idx=False):
        """
        plot sed fitting and redshift pdf at same figure\n
        idx: int
          search by index
        lrange: array_like
          decide the lower and upper limit of wavelength display range
        log_wavelen: bool
          display the wavelength axis in log scale
        id_is_idx: bool
          search by id instead of index
        individual_templates: bool
          display all the SED in sed template sets
        """
        if id_is_idx:
            idx = self.id_to_idx(idx)
        plt.figure(figsize=(7.5,3), dpi=120)
        ax_sed = plt.axes([0.10, 0.18, 0.45, 0.8]) #[left, bottom, width, height]
        ax_pdf = plt.axes([0.65, 0.18, 0.33, 0.8])
        ax_sed = self.show_sed_fitting(idx, individual_templates=individual_templates, 
                                        log_wavelen=log_wavelen, lrange=lrange, ax=ax_sed)
        ax_pdf = self.show_pdf(idx, ax=ax_pdf)
        
    def show_pdf(self, idx, zmax=6, id_is_idx=False, ax=False):
        """
        plot redshift pdf only\n
        idx: int
          search by index
        id_is_idx:
          search by id instead of index
        """
        if id_is_idx:
            idx = self.id_to_idx(idx)
        ttlo = False    # plt.tight_layout()
        if not ax:
            ttlo = True
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111)
        self.get_pdf()
        ax.plot(self.zgrid, self.pzout[idx], c='orange', alpha=.8, lw=1)
        ax.fill_between(self.zgrid, self.pzout[idx], np.zeros(self.zgrid.size), color='yellow', alpha=.8)
        ax.plot([self.z_best[idx],self.z_best[idx]], [0,120], color='C0', 
                linewidth=1, label='$z_{best}$', alpha=0.8)
        ax.plot([self.z_spec[idx],self.z_spec[idx]], [0,120], color='C3', 
                linewidth=1, label='$z_{spec}$', alpha=0.8)
        ax.set_xlim(0, zmax)
        ax.set_ylim(0, 1.1*max(self.pzout[idx]))
        ax.set_xlabel('z')
        ax.set_ylabel('p(z)')
        ax.legend(loc='upper right')
        
        if ttlo:
            plt.tight_layout()
        
        return ax

    def show_sed_fitting(self, idx, lrange=[2000, 11000], log_wavelen=False, 
                         individual_templates=False, id_is_idx=False, ax=False):
        """
        plot sed fitting only\n
        # Parameters\n
        idx: int
          search by index
        lrange: array_like
          decide the lower and upper limit of wavelength display range
        log_wavelen: bool
          display the wavelength axis in log scale
        id_is_idx: bool
          search by id instead of index
        individual_templates: bool
          display all the SED in sed template sets
        """
        if id_is_idx:
            idx = self.id_to_idx(idx)
        ttlo = False    # plt.tight_layout()
        if not ax:
            ttlo = True
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111)
        self.get_eazy_sed(idx, individual_templates=individual_templates)
        if individual_templates:
            ax.plot(self.lambdaz, self.itemp_sed.sum(axis=1), lw=1.5, alpha=.8, zorder=-1, c='k')
            ax.plot(self.lambdaz, self.itemp_sed.sum(axis=1), lw=1, alpha=1, zorder=-1, c='C0')
            ax.plot(self.lambdaz, self.itemp_sed, lw=1, alpha=.5, zorder=-1, c='C0')
        else:
            ax.plot(self.lambdaz, self.itemp_sed, lw=1, alpha=.8, zorder=-1, c='C0')
        ax.scatter(self.lci, self.obs_sed, c='white', marker='o', s=35,  alpha=1)
        ax.scatter(self.lci, self.obs_sed, c='C3', marker='o', s=20,  alpha=.8)

        highsnr = self.fobs/self.efobs > 2
        ax.errorbar(self.lci[highsnr], self.fobs[highsnr], yerr=self.efobs[highsnr], ecolor='k', color='k',
                    fmt='o', alpha=.8, markeredgecolor='k', markerfacecolor='None', 
                    markeredgewidth=1, elinewidth=1, ms=5, zorder=2)
        ax.errorbar(self.lci[~highsnr], self.fobs[~highsnr], yerr=self.efobs[~highsnr], ecolor='0.7', 
                    color='0.7',fmt='o', alpha=.5, markeredgecolor='0.7', markerfacecolor='None', 
                    markeredgewidth=1, elinewidth=1, ms=5, zorder=1)
        if log_wavelen:
            ax.semilogx()
        ax.set_xlim(lrange[0], lrange[1])
        ax.set_ylim(-0.05*max(self.obs_sed),1.05*max((max(self.obs_sed), max(self.fobs+self.efobs))))
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('$f_{\lambda}$')
        
        if ttlo:
            plt.tight_layout()
        
        return ax
        
    def on_pick(self, event):

        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        if x is None or y is None:
            pass
        else:
            distances = np.hypot(x - self.z_spec[event.ind], y - self.z_best[event.ind])
            idx_min = distances.argmin()
            dataidx = event.ind[idx_min]
            self.selected_idx = dataidx
            self.update()

    def update(self):

        # idx for total
        _idx = self.idx[self.selected_idx]
        # z_phot-z_spec figure
        self.selected_fg.set_visible(True)
        self.selected_bg.set_visible(True)
        self.selected_fg.set_data(self.z_spec[_idx], self.z_best[_idx])
        self.selected_bg.set_data(self.z_spec[_idx], self.z_best[_idx])
        # update selected info
        self.selected_info.set_text('{:>9d}\n{:>9d}\n{:>9.4f}\n{:>9.4f}\n{:>9.4f}'.format(self.id[_idx], _idx, self.z_spec[_idx], self.z_best[_idx], self.qz[_idx]))
        if self.combo == 1:
            self.fitting_info.set_text('{:<9d}\n{:<9.4f}'.format(self.temp[_idx], self.chi2[_idx]))
        elif self.combo == 2:
            self.fitting_info.set_text('{:<3d}and{:>3d}\n{:<9.4f}'.format(self.temp[0][_idx], self.temp[1][_idx], self.chi2[_idx]))
        else:
            self.fitting_info.set_text('{:<20}\n{:<9.4f}'.format('---', self.chi2[_idx]))
        # sed fitting figure
        self.ax_sed.cla()
        self.ax_sed = self.show_sed_fitting(_idx, ax=self.ax_sed, lrange=self.lrange, 
                                            log_wavelen=self.log_wavelen, 
                                            individual_templates=self.individual_temp)
        # PDF figure
        self.ax_pdf.cla()
        self.ax_pdf = self.show_pdf(_idx, ax=self.ax_pdf, zmax=self.zmax)
        self.fig.canvas.draw()
        
    def get_eazy_sed(self, idx, mag_zp=-48.6, scale_flambda=1.e-17, 
                     individual_templates=False):

        # load necessary parameters
        tempfilt = self.tempfilt['tempfilt']
        fnu = self.tempfilt['fnu']
        efnu = self.tempfilt['efnu']
        self.lci = self.tempfilt['lc']
        zgrid = self.tempfilt['zgrid']
        templam = self.temp_sed['templam']
        tempseds = self.temp_sed['temp_seds']
        izbest = self.coeffs['izbest']
        coeffs = self.coeffs['coeffs']
        da = self.temp_sed['da']
        db = self.temp_sed['db']
        zi = zgrid[izbest[idx]]
        # redshift wavelength grid
        self.lambdaz = (1+zi) * templam
        if scale_flambda:
            flambda_coeff = 10**(-0.4*(mag_zp+48.6))*3.e18/scale_flambda
        else:
            flambda_coeff = 5500**2
        # flux derived by SED templates
        self.obs_sed = np.dot(tempfilt[izbest[idx]], coeffs[idx])/self.lci**2*flambda_coeff
        # templates
        self.itemp_sed = np.dot(tempseds, coeffs[idx])
        if individual_templates:
            self.itemp_sed = tempseds*coeffs[idx]
        self.itemp_sed /= (1+zi)**2
        self.itemp_sed *= (1/5500)**2*flambda_coeff
        # convert observed f_nu (from the catalog) to f_lambda
        self.fobs = fnu[idx]/self.lci**2*flambda_coeff
        self.efobs = efnu[idx]/self.lci**2*flambda_coeff
        # IGM correction for SED
        lim1 = np.where(templam < 912)
        lim2 = np.where((templam >= 912) & (templam < 1026))
        lim3 = np.where((templam >= 1026) & (templam < 1216))
        if lim1[0].size > 0: 
            self.itemp_sed[lim1] *= 0.
        if lim2[0].size > 0: 
            self.itemp_sed[lim2] *= 1.-db[izbest[idx]]
        if lim3[0].size > 0: 
            self.itemp_sed[lim3] *= 1.-da[izbest[idx]]    


    def get_pdf(self):

        NOBJ = self.photoz_info['NOBJ']
        NZ = self.photoz_info['NZ']
        NK = self.pz['NK']
        if NK:
            kidx = self.pz['kidx']
            idx = kidx >= NK
            kidx[idx] = 0
            priorzk = self.pz['priorzk']
            priorz = priorzk[kidx,:]
            priorz[idx] = np.ones(NZ)
        else:
            priorz = np.ones([NOBJ, NZ])
        coeff = np.ones(NZ)    
        chi2fit = self.pz['chi2fit']
        self.zgrid = self.tempfilt['zgrid']
        # convert chi2fit to p(z)
        min_chi2fit = np.multiply(np.min(chi2fit, axis=1).reshape(NOBJ,1), coeff)
        self.pzout = np.exp(-0.5*(chi2fit-min_chi2fit)) * priorz
        self.pzout /= np.multiply(np.trapz(self.pzout,self.zgrid,axis=1).reshape(NOBJ,1), coeff)


    def id_to_idx(self, idx):

        id_idx = idx
        id_err_flag = False
        _idx = np.where(self.id==id_idx)[0]
        try:
            idx = _idx[0]
        except IndexError:
            print('id %d not found'%(id_idx))
            id_err_flag = True
        if id_err_flag:
            raise IndexError
        if _idx.shape[0] > 1:
            print('duplicated id %d, index:'%(id_idx),idx[0])
            id_err_flag = True
        if id_err_flag:
            raise IndexError
        
        return idx


def nMAD(arr):
    """
    Get the nMAD statistic of the input array, where
    nMAD = 1.48 * median(abs(arr) - median(arr)).
    """
    return 1.48 * np.median( np.abs(arr - np.median(arr)) )


def zout_analysis(z_spec, z_best):
    """
    Return dz, fraction of outliers and sigma
    """
    idx = z_best > 0
    dz = (z_best[idx]-z_spec[idx]) / (1+z_spec[idx])
    idx_outliers = np.fabs(dz) >= 0.15
    f_out = idx_outliers.sum() / idx.sum() * 100
    sigma = nMAD(dz)

    return dz, f_out, sigma


def read_eazy_param(param_file):
    """
    Return parameters dict, note that all parameters are in format of string!
    """
    param_dict = {}
    # read parameter file
    with open(param_file, 'r') as params:
        for line in params:
            # remove white space
            param = []
            line_ = line.replace('\n', '').split(' ')
            param_line = [i for i in line_  if i]
            # remove empty line
            if param_line:
                for p in param_line:
                    # remove comments after '#'
                    if '#' in p:
                        break
                    param.append(p)
            if param:
                param_dict[param[0]] = param[1]
    try:
        has_bin_out = param_dict['BINARY_OUTPUT']
    # check for key parameter
    # BINARY_OUTPUT
    except KeyError:
        param_dict['BINARY_OUTPUT'] = 0
    else:
        if has_bin_out in ['y', 'Y', '1']:
            param_dict['BINARY_OUTPUT'] = 1
        else:
            param_dict['BINARY_OUTPUT'] = 0
    
    return param_dict


def write_eazy_param(param_dict, name):
    """
    Save the parameter dict to file
    """
    with open(name, 'w+') as output_parameter:
        for p_name, p_value in zip(param_dict.keys(), param_dict.values()):
            output_parameter.write('{:<22}{:<}\n'.format(p_name, p_value))


def read_eazy_binary(output_dir='', output_file='', cache_file=None):
    """
    Read the binary output of eazy
    # Example
    >>> tempfilt, coeffs, temp_sed, pz, photoz_info = read_eazy_binary(output_dir='output', output_file='photoz')
    """
    # get path of the output files
    path = os.path.join(output_dir, output_file)
    pz_path = path + '.pz'
    coeff_path = path + '.coeff'
    if cache_file:
        tempfilt_path = os.path.join(output_dir, cache_file)
    else:
        tempfilt_path = path + '.tempfilt'
    temp_sed_path = path + '.temp_sed'
    ## .tepfilt file
    with open(tempfilt_path, 'rb') as tempfilt_file:
        # tempfilt: fluxes derievd from templates at different bandpass
        # lc      : mean wavelength of each filters
        # zgrid   : redshift grid
        # fnu     : flux of galaxies in unit of erg s-1 cm-1 Hz-1
        # efnu    : flux error of galaxies
        tempfilt_info = np.fromfile(file=tempfilt_file, dtype=np.int32, count=4)
        NFILT = tempfilt_info[0]
        NTEMP = tempfilt_info[1]
        NZ = tempfilt_info[2]
        NOBJ = tempfilt_info[3]
        tempfilt = np.fromfile(file=tempfilt_file, dtype=np.double, 
                    count=NFILT*NTEMP*NZ).reshape((NZ, NTEMP, NFILT)).transpose(0,2,1)
        lc = np.fromfile(file=tempfilt_file, dtype=np.double, count=NFILT)
        zgrid = np.fromfile(file=tempfilt_file, dtype=np.double, count=NZ)
        fnu = np.fromfile(file=tempfilt_file, dtype=np.double, 
                            count=NFILT*NOBJ).reshape((NOBJ, NFILT))
        efnu = np.fromfile(file=tempfilt_file, dtype=np.double, 
                            count=NFILT*NOBJ).reshape((NOBJ, NFILT))
    tempfilt = {'tempfilt':tempfilt, 'lc':lc, 'zgrid':zgrid, 'fnu':fnu, 'efnu':efnu}
    ## .coeff file
    with open(coeff_path, 'rb') as coeff_file:
        # .coeff file contains coefficients of templates
        # coeffs: coefficients of each template for each object
        # izbest: the index of the best redshift in zgrid
        # tnorm : normalize coefficient for the templates?
        coeff_info = np.fromfile(file=coeff_file, dtype=np.int32, count=4)
        NFILT = coeff_info[0]
        NTEMP = coeff_info[1]
        NZ = coeff_info[2]
        NOBJ = coeff_info[3]
        coeffs = np.fromfile(file=coeff_file, dtype=np.double, 
                                count=NTEMP*NOBJ).reshape((NOBJ,NTEMP))
        izbest = np.fromfile(file=coeff_file, dtype=np.int32,  count=NOBJ)
        tnorm = np.fromfile(file=coeff_file, dtype=np.double, count=NTEMP)
        coeffs = {'coeffs':coeffs, 'izbest':izbest, 'tnorm':tnorm}
    ## .temp_sed file
    with open(temp_sed_path, 'rb') as temp_sed_file:
        # NTEMPL   : length of template wavelength array
        # templam  : templates wavelength array
        # temp_seds: flambda of all the templates
        # da       : Da
        # db       : Db
        temp_sed_info = np.fromfile(file=temp_sed_file, dtype=np.int32, count=3)
        NTEMPL = temp_sed_info[1]
        templam = np.fromfile(file=temp_sed_file, dtype=np.double, count=NTEMPL)
        temp_seds = np.fromfile(file=temp_sed_file, dtype=np.double, 
                                count=NTEMPL*NTEMP).reshape((NTEMP,NTEMPL)).T
        da = np.fromfile(file=temp_sed_file, dtype=np.double, count=NZ)
        db = np.fromfile(file=temp_sed_file, dtype=np.double, count=NZ)
    temp_sed = {'templam':templam, 'temp_seds':temp_seds, 'da':da, 'db':db}
    ## .pz_file
    with open(pz_path, 'rb') as pz_file:
        # NK     : number of prior's mag bins
        # kbins  : number of pz's z bins
        # priorzk: prior curves of different bins
        pz_info = np.fromfile(file=pz_file, dtype=np.int32, count=2)
        NZ = pz_info[0]
        NOBJ = pz_info[1]
        chi2fit = np.fromfile(file=pz_file, dtype=np.double, 
                                count=NZ*NOBJ).reshape((NOBJ,NZ))
        nk = np.fromfile(file=pz_file,dtype=np.int32, count=1)
        if len(nk) > 0:
            NK = nk[0]
            kbins = np.fromfile(file=pz_file, dtype=np.double, count=NK)
            priorzk = np.fromfile(file=pz_file, dtype=np.double, 
                                    count=NZ*NK).reshape((NK,NZ))
            kidx = np.fromfile(file=pz_file, dtype=np.int32,  count=NOBJ)
            pz = {'chi2fit':chi2fit, 'kbins':kbins, 'priorzk':priorzk, 'kidx':kidx, 'NK':NK}
        else:
            pz = {'chi2fit':chi2fit, 'kbins':None, 'priorzk':None, 'kidx':None, 'NK':None}
    photoz_info = {'NFILT':NFILT, 'NTEMP':NTEMP, 'NTEMPL':NTEMPL, 'NZ':NZ, 'NOBJ':NOBJ}

    return tempfilt, coeffs, temp_sed, pz, photoz_info


if __name__ == '__main__':
    p = read_eazy_param('C:/work/template_calibration/20220725/eazy-demo/zphot.param')
    write_eazy_param(p, 'C:/work/template_calibration/20220725/eazy-demo/zphot_.param')
    print(p)
