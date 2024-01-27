#==============================================================================
#                                 IMPORTS
#==============================================================================
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.animation as animation

from datetime import datetime


#==============================================================================
#                          GENERAL AUXILARY PLOTTING FUNCTIONS
#==============================================================================
class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
        # Formats axis to 1 decimal place
        self.format = "%1.1f"


class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format
        

def colorbar_fmt(x, pos):
    # Formats colorbar to 0 decimal place
    return '{:.0f}'.format(float(x))


def sci_not(number, exponent=None):
    """
    Convert number to a scientific notation-like format
    
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with fixed number of significant
    decimal digits. The exponent to be used can also be specified
    explicitly.
    
    Args:
        number (str or foat): The number to convert. If str, the string will be returend. 
        exponent (None or int): None uses scientific nortation, int allows exponent to be set
        
    Returns:
        (str): output string
    """
    
    if isinstance(number, str):
        return f"{number}"
    
    if exponent is None:
        if number == 0:
            exponent = 1
        else:
            exponent = int(math.floor(math.log10(abs(number))))
        
    coeff = number / float(10**exponent)
    
    #set precsion in return line
    return r"${:.3f}\cdot10^{{{:d}}}$".format(coeff, exponent)


def approx_arc_len(mfep_cv):
    """
    Normalizes a path by approximate arc lenght
    
    Args:
        mfep_cv (np.ndarray): the CV location of the path
        
    Returns:
        arc_lens (np.ndarray): normalized path length
    """
    
    arc_lens = np.zeros(mfep_cv.shape[0])
    for i in range(1, arc_lens.shape[0]):
        arc_lens[i] = np.linalg.norm(mfep_cv[i] - mfep_cv[i-1])
    arc_lens = np.cumsum(arc_lens)
    arc_lens = arc_lens / np.max(arc_lens)
    return arc_lens


#==============================================================================
#                          MAIN PLOTTING FUNCTION
#==============================================================================
def plot_fes(fes, options=dict() ):
    """
    Plots the FES with desired options
    
    Args:
        fes (fes class instance): the ffes to plot
        options (dict, optional): options to make the graph look nice
            - dpi (int): dots per square inch of figure
            - cv_labels (list): contains strings of the CV names
            - time_unit (str): time unit to display for tracking over time
            - time_str (str): What you wish to call the time axis
            - time_vals (list): the time values for each FES data frame
            - x_axis_exponent (int): the number to set as base 10 exponent for time axis
            - title (str): plot title
            - energy_unit_str (str): string for idenitfying energy unit
            - track (bool): whether to track plot over time
            - gif (bool): whether to animate a plot
            - save_file (str): filename handle (no extension) to save png (and potentially gif to)
            - fps (int): frames per second for gif
            - plot_locs (bool): whether to plot locations of transiiton state and minima
            - project_1d (bool); wether to project 2 dimension FES into 1d (required for 3+)
            - scatter_size (bool): point size for minima marks
            - scatter_dagger_size: point size for transiton state dagger marks
            - 2d_spacing (float): color bar level spacing 
            - 2d_tick_skip (int): how many ticks to skip on color bar
            - skip_path_nodes (int): how many points to skip when plotting path
            - dagger_lift (float): how much to raise/lower transition state dagger location by
    
    Returns:
        fig (matplotlib figure): output graph
        axs (matplotlib axes): output axes
    
    """
    
    if "dpi" not in options:
        options["dpi"] = 300
    
    if "cv_labels" not in options:
        options["cv_labels"] = fes.cv_cols
        
    if "time_unit" not in options:
        options["time_unit"] = ""

    if "time_str" not in options:
        options["time_str"] = "Time"
    
    if "time_vals" not in options:
        options["time_vals"] = list(range(fes.num_fes))
    else:
        if len(options["time_vals"]) != fes.num_fes:
            raise ValueError(f"time_vals is not of correct length: {fes.num_fes}")
    
    if "x_axis_exponent"not in options:
        options["x_axis_exponent"] = 3
    
    if "title" not in options:
        options["title"] = None
    
    if "energy_unit_str" not in options:
        options["energy_unit_str"] = ""

    if fes.num_fes == 1:
        options["track"] = False
    
    elif not "track" in options:
        if fes.num_fes > 1:
            options["track"] = True
        else:
            options["track"] = False
    
    if not "gif" in options:
        if options["track"]:
            options["gif"] = True
        else:
            options["gif"] = False
        
    if not "save_file" in options:
        options["save_file"] = ""
        
    if "fps" not in options:
        options["fps"] = 4
    
    if "plot_locs" not in options:
        options["plot_locs"] = True
    
    if fes.dims > 1:
        if fes.dims == 2:
            if "project_1d" not in options:
                options["project_1d"] = False
        else:
            options["project_1d"] = True
    else:
        options["project_1d"] = False
            
    if "scatter_size" not in options:
        options["scatter_size"] = 5
    
    if "scatter_dagger_size" not in options:
        options["scatter_dagger_size"] = 2 * options["scatter_size"]
        
    if not "2d_spacing" in options:
        options["2d_spacing"] = 5
    
    if not "2d_tick_skip" in options:
        options["2d_tick_skip"] = 2
    
    if not "skip_path_nodes" in options:
        options["skip_path_nodes"] = 1  
    
    if not "dagger_lift" in options:
        options["dagger_lift"] = 3
    
    
    plt.rcParams['font.size']       = 10
    plt.rcParams['axes.titlesize']  = 10
    plt.rcParams['axes.labelsize']  = 10 
    plt.rcParams['legend.fontsize'] = 10
    
    plt.rcParams["font.family"]      = 'DejaVu Sans'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm']      = 'DejaVu Sans'
    plt.rcParams['mathtext.it']      = 'DejaVu Sans:italic'
    plt.rcParams['mathtext.bf']      = 'DejaVu Sans:bold'

    if options["track"]:
        options["figsize"] = (7, 3.5)
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, animated=True, figsize=options["figsize"])
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=None)
        
        axs[1].set_xlabel(options["cv_labels"][0])
        if options["time_unit"] == "":
            axs[1].set_xlabel(options["time_str"])
        else:
            axs[1].set_xlabel(options["time_str"] + " (" + options["time_unit"] + ")" )
        axs[1].set_ylabel("Free Energy (" + options["energy_unit_str"] + ")")
        
        axs[1].xaxis.set_major_formatter(OOMFormatter(0, "%1.0f"))
        fig.align_xlabels() # make sure x axis lables are aligned
        
        # Calculate & Setup the tracking limits
        track_buffer = 3
        fes.results["ts_rev"] = fes.results["ts_val"] - fes.results["dG"]
        min_energy_val = np.min(fes.results["minima_vals"])
        max_energy_val = np.max( np.concatenate([fes.results["ts_val"], fes.results["ts_rev"]] ) )
        axs[1].set_ylim(min_energy_val - track_buffer, max_energy_val + track_buffer)
        axs[1].set_xlim(np.min(np.asarray(options["time_vals"])), np.max(np.asarray(options["time_vals"])) )
        
        x_test_format = OOMFormatter(options["x_axis_exponent"], "%1.0f")
        axs[1].xaxis.set_major_formatter(x_test_format)

    else:
        options["figsize"] = (3.25, 3.5)
        fig, ax = plt.subplots(1, 1, animated=True, figsize=options["figsize"])
        axs = [ax]
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    
    fig.suptitle(options["title"])

    if options["project_1d"] or fes.dims == 1:
        axs[0].set_ylabel(r'$\mathrm{\Delta}$G' + " (" + options["energy_unit_str"] + ")")
        if options["project_1d"]:
            axs[0].set_xlabel("Normalized Path Length")
        else:
            axs[0].set_xlabel(options["cv_labels"][0])
    else:
        axs[0].set_ylabel(options["cv_labels"][1])

    # --------------- Specific Plotting Flowchart ------------------------------
    if (fes.dims == 1) or (options["project_1d"] == True):
    
        zero_line = axs[0].axhline(0.0, color="gray", linestyle="dashed", alpha=0.3, zorder=-1, animated=True)
        
        if options["track"] and options["gif"]:
            #Left graph, text annotation
            annotation_box_details = dict(boxstyle='round', facecolor='wheat')
                    
            #Left graph, FES
            fes_plot, = axs[0].plot([], [], color="blue", animated=True, zorder=0)
            
            #Left graph, prodict/reactant wells and TS location
            if options["plot_locs"]:
                ts_dagger, = axs[0].plot([], [], c="black", marker=r"$\ddag$", markersize=options["scatter_dagger_size"],
                    linestyle="", animated=True, zorder=20)
                react_prod_dot, = axs[0].plot([], [], c="black", marker="o", markersize=options["scatter_size"],
                    linestyle="", animated=True, zorder=20)
            
            annotation = axs[0].text(0.017, 0.9875, sci_not("") + options["time_unit"], transform=axs[0].transAxes, fontsize=10,\
                ha='left', va='top', \
                fontweight="normal", animated=True, bbox=annotation_box_details, zorder=20)
        
            #now for right graph
            ts_line,     = axs[1].plot([], [], c="red", label=r"$\mathrm{\Delta {G^{\ddag}}_{f}}$", animated=True)
            ts_rev_line, = axs[1].plot([], [], c="purple", label=r"$\mathrm{\Delta {G^{\ddag}}_{r}}$", animated=True)
            dG_line,     = axs[1].plot([], [], c="blue", label=r"$\mathrm{\Delta G}$", animated=True)
            # make sure legend is displayed for right graph
            axs[1].legend(loc=2, ncols=1) #loc=2 is upper left, 0 is best
            
            if not options["project_1d"]:
                pmf_xmin = np.min( np.concatenate(fes.results["cv_vectors"]) )
                pmf_xmax = np.max( np.concatenate(fes.results["cv_vectors"]) )
                all_fes = np.concatenate(fes.results["fes"])
                pmf_ymax = np.max( all_fes[np.isfinite(all_fes)] )
                pmf_ymin = np.min( all_fes[np.isfinite(all_fes)] )
            
            else:
                pmf_xmin = 0.0
                pmf_xmax  = 1.0
                all_mfep = np.concatenate([fes.results["mfep"][x][1] for x in range(len(fes.results["mfep"]))] )
                pmf_ymax = np.max( all_mfep[np.isfinite(all_mfep)] )
                pmf_ymin = np.min( all_mfep[np.isfinite(all_mfep)] )
            
            axs[0].set_ylim(pmf_ymin - track_buffer, pmf_ymax + track_buffer)
            axs[0].set_xlim(pmf_xmin, pmf_xmax)
            
            ani = animation.FuncAnimation(fig, animate_1d, frames=fes.num_fes, interval=1000, blit=True,
                fargs=(annotation, fes_plot, react_prod_dot, ts_dagger, ts_line, ts_rev_line, dG_line, fes, options))
        
        else:
            if options["plot_locs"]:
                if (fes.dims == 1):
                    fes_plot, = axs[0].plot(fes.results["cv_vectors"][-1][0], fes.results["fes"][-1], color="blue", animated=True, zorder=0)
                    ts_dagger, = axs[0].plot(fes.results["ts_loc"][-1], fes.results["ts_val"][-1] + options["dagger_lift"], c="black", marker=r"$\ddag$", markersize=options["scatter_dagger_size"],
                        linestyle="", animated=True, zorder=20)
                    react_prod_dot, = axs[0].plot(fes.results["minima_locs"][-1], fes.results["minima_vals"][-1], c="black", marker="o", markersize=options["scatter_size"],
                        linestyle="", animated=True, zorder=20)
                else:
                    arc_locs = approx_arc_len(fes.results["mfep"][-1][0])
                    fes_plot, = axs[0].plot(arc_locs, fes.results["mfep"][-1][1], color="blue", zorder=0)
                    ts_arc = arc_locs[ np.argwhere(fes.results["mfep"][-1][1] == fes.results["ts_val"][-1]) ]
                    ts_dagger, = axs[0].plot(ts_arc, fes.results["ts_val"][-1] + options["dagger_lift"], c="black", marker=r"$\ddag$", markersize=options["scatter_dagger_size"],
                        linestyle="", animated=True, zorder=20)
                    minima_arc = arc_locs[ np.argwhere(np.in1d(fes.results["mfep"][-1][1], fes.results["minima_vals"][-1]) ) ]
                    react_prod_dot, = axs[0].plot(minima_arc, np.asarray(fes.results["minima_vals"][-1]), c="black", marker="o", markersize=options["scatter_size"],
                        linestyle="", animated=True, zorder=20)

            if options["track"]:
                ts_line,     = axs[1].plot(options["time_vals"], fes.results["ts_val"], c="red", label=r"$\mathrm{\Delta {G^{\ddag}}_{f}}$")
                ts_rev_line, = axs[1].plot(options["time_vals"], fes.results["ts_rev"], c="purple", label=r"$\mathrm{\Delta {G^{\ddag}}_{r}}$")
                dG_line,     = axs[1].plot(options["time_vals"], fes.results["dG"], c="blue", label=r"$\mathrm{\Delta G}$")
                axs[1].legend(loc=2, ncols=1) #loc=2 is upper left, 0 is best
    
    
    # --------------- Specific Plotting Flowchart ------------------------------
    # the 2 dimensional case is unique
    else:

        pmf_xmin = np.min( np.concatenate([fes.results["cv_vectors"][i][0] for i in range(len(fes.results["cv_vectors"]))] ) )
        pmf_xmax = np.max( np.concatenate([fes.results["cv_vectors"][i][0] for i in range(len(fes.results["cv_vectors"]))] ) )
        
        pmf_ymin = np.min( np.concatenate([fes.results["cv_vectors"][i][1] for i in range(len(fes.results["cv_vectors"]))] ) )
        pmf_ymax = np.max( np.concatenate([fes.results["cv_vectors"][i][1] for i in range(len(fes.results["cv_vectors"]))] ) )
        
        all_fes = np.concatenate(fes.results["fes"])
        max_Z = math.ceil( np.max(all_fes[np.isfinite(all_fes)]) )
        levels = np.linspace(0.0, max_Z, num=int((max_Z)/options["2d_spacing"])+1, endpoint=True)
        
        emptyZ = np.full_like(fes.results["fes"][-1], np.nan)

        cbar_levels = list( levels[::options["2d_tick_skip"]] )
            
        if options["track"] and options["gif"]:
            #Left graph, text annotation
            annotation_box_details = dict(boxstyle='round', facecolor='wheat')
                    
            #Left graph, FES
            fes_plot = axs[0].contourf(fes.results["cv_vectors"][-1][0], fes.results["cv_vectors"][-1][1], emptyZ, alpha=1, cmap='jet', levels=levels, zorder=0)
            cbar = fig.colorbar(fes_plot, ax=axs[0], format=ticker.FuncFormatter(colorbar_fmt), ticks=cbar_levels)
                    
            #Left graph, prodict/reactant wells and TS location
            if options["plot_locs"]:
                ts_dagger, = axs[0].plot([], [], c="black", marker=r"$\ddag$", markersize=options["scatter_dagger_size"],
                    linestyle="", animated=True, zorder=20)
                react_prod_dot, = axs[0].plot([], [], c="black", marker="o", markersize=options["scatter_size"],
                    linestyle="", animated=True, zorder=20)
                path_trace, = axs[0].plot([], [], linestyle="dashed", color="grey", animated=True, zorder=10)

            
            annotation = axs[0].text(0.017, 0.9875, sci_not("") + options["time_unit"], transform=axs[0].transAxes, fontsize=10,\
                ha='left', va='top', \
                fontweight="normal", animated=True, bbox=annotation_box_details, zorder=20)
        
            #now for right graph
            ts_line,     = axs[1].plot([], [], c="red", label=r"$\mathrm{\Delta {G^{\ddag}}_{f}}$", animated=True)
            ts_rev_line, = axs[1].plot([], [], c="purple", label=r"$\mathrm{\Delta {G^{\ddag}}_{r}}$", animated=True)
            dG_line,     = axs[1].plot([], [], c="blue", label=r"$\mathrm{\Delta G}$", animated=True)
            
            # make sure legend is displayed for right graph
            axs[1].legend(loc=2, ncols=1) #loc=2 is upper left, 0 is best
            
            axs[0].set_ylim(pmf_ymin, pmf_ymax)
            axs[0].set_xlim(pmf_xmin, pmf_xmax)
            
            ani = animation.FuncAnimation(fig, animate_2d, frames=fes.num_fes, interval=1000, blit=True,
                fargs=(annotation, fes_plot, react_prod_dot, ts_dagger, ts_line, ts_rev_line, dG_line, path_trace, fes, options, axs, levels))
        
        else:
            #Left graph, text annotation

            #Left graph, FES
            fes_plot = axs[0].contourf(fes.results["cv_vectors"][-1][0], fes.results["cv_vectors"][-1][1], fes.results["fes"][-1], alpha=1, cmap='jet', levels=levels, zorder=0)
            cbar = fig.colorbar(fes_plot, ax=axs[0], format=ticker.FuncFormatter(colorbar_fmt), ticks=cbar_levels)
            
            #Left graph, prodict/reactant wells and TS location
            if options["plot_locs"]:
            
                ts_dagger, = axs[0].plot(fes.results["ts_loc"][-1][1], fes.results["ts_loc"][-1][0], c="black", marker=r"$\ddag$", markersize=options["scatter_dagger_size"],
                    linestyle="", animated=True, zorder=20)
                minima_list_x = [fes.results["minima_locs"][-1][0][0],fes.results["minima_locs"][-1][1][0]]
                minima_list_y = [fes.results["minima_locs"][-1][0][1],fes.results["minima_locs"][-1][1][1]]
                react_prod_dot, = axs[0].plot(minima_list_x, minima_list_y, c="black", marker="o", markersize=options["scatter_size"],
                    linestyle="", animated=True, zorder=20)
                
                xpath = fes.results["mfep"][-1][0][:,1][::options["skip_path_nodes"]]
                ypath = fes.results["mfep"][-1][0][:,0][::options["skip_path_nodes"]]
                if options["skip_path_nodes"] != 1:
                    xpath[-1] = fes.results["mfep"][-1][0][-1,1]
                    ypath[-1] = fes.results["mfep"][-1][0][-1,0]
                path_trace, = axs[0].plot(xpath, ypath, linestyle="dashed", color="grey", animated=True, zorder=10)
        
            if options["track"]:
                ts_line,     = axs[1].plot(options["time_vals"], fes.results["ts_val"], c="red", label=r"$\mathrm{\Delta {G^{\ddag}}_{f}}$")
                ts_rev_line, = axs[1].plot(options["time_vals"], fes.results["ts_rev"], c="purple", label=r"$\mathrm{\Delta {G^{\ddag}}_{r}}$")
                dG_line,     = axs[1].plot(options["time_vals"], fes.results["dG"], c="blue", label=r"$\mathrm{\Delta G}$")
    
                # make sure legend is displayed for right graph
                axs[1].legend(loc=2, ncols=1) #loc=2 is upper left, 0 is best


    if options["save_file"]:
        
        metadata = {}
        metadata["comment"] = "Created with Matplotlib & fes_tools package"
        now = datetime.now()
        metadata["date"] = now.strftime("%d/%m/%Y %H:%M:%S")
        
        if options["track"] and options["gif"]:
            gif_writer = animation.ImageMagickWriter(fps=options["fps"], bitrate=-1, metadata=metadata)
            ani.save(f'{options["save_file"]}.gif', writer=gif_writer)
        
        fig.savefig(f'{options["save_file"]}.png', dpi=options["dpi"], metadata=metadata)
    
    return fig, axs
        

#===============================================================================
#                       INNER ANIMATION FUNCTIONS
#===============================================================================
def animate_1d(i, annotation, fes_plot, react_prod_dot, ts_dagger, ts_line, ts_rev_line, dG_line, fes, options):
    
    #these are cumulative graphs, so need to keep "history"/past locations
    #i+1 bc list indexing [:a] does not include a
    
    current_time  = options["time_vals"][:i+1]
    
    annotation.set_text(sci_not(current_time[-1]) +  " " + options["time_unit"])
    ts_line.set_data(current_time, fes.results["ts_val"][:i+1])
    ts_rev_line.set_data(current_time, fes.results["ts_rev"][:i+1])
    dG_line.set_data(current_time, fes.results["dG"][:i+1])
    
    #Plot the relevant locations of the PMF on the left subplot
    if not options["project_1d"]:
        fes_plot.set_data(fes.results["cv_vectors"][i][0], fes.results["fes"][i])
    else:
        arc_locs = approx_arc_len(fes.results["mfep"][i][0])
        fes_plot.set_data(arc_locs, fes.results["mfep"][i][1])
        
    if options["plot_locs"] == True:
        if options["project_1d"]:
            ts_arc = arc_locs[ np.argwhere(fes.results["mfep"][i][1] == fes.results["ts_val"][i]) ]
            minima_arc = arc_locs[ np.argwhere(np.in1d(fes.results["mfep"][i][1],fes.results["minima_vals"][i]) ) ]
            ts_dagger.set_data([ts_arc, fes.results["ts_val"][i] + options["dagger_lift"]])
            react_prod_dot.set_data([minima_arc, fes.results["minima_vals"][i]])
        else:
            ts_dagger.set_data([fes.results["ts_loc"][i], fes.results["ts_val"][i] + options["dagger_lift"]])
            react_prod_dot.set_data([fes.results["minima_locs"][i], fes.results["minima_vals"][i]])
        
    return annotation, fes_plot, react_prod_dot, ts_dagger, ts_line, ts_rev_line, dG_line
    
    
def animate_2d(i, annotation, fes_plot, react_prod_dot, ts_dagger, ts_line, ts_rev_line, dG_line, path_trace, fes, options, axs, levels):

    #these are cumulative graphs, so need to keep "history"/past locations
    #i+1 bc list indexing [:a] does not include a
    current_time  = options["time_vals"][:i+1]
    
    annotation.set_text(sci_not(current_time[-1]) +  " " + options["time_unit"])
    ts_line.set_data(current_time, fes.results["ts_val"][:i+1])
    ts_rev_line.set_data(current_time, fes.results["ts_rev"][:i+1])
    dG_line.set_data(current_time, fes.results["dG"][:i+1])
    
    #Plot the relevant locations of the PMF on the left subplot
    fes_plot = axs[0].contourf(fes.results["cv_vectors"][i][0], fes.results["cv_vectors"][i][1], fes.results["fes"][i], alpha=1, cmap='jet', levels=levels, zorder=0)
    
    if options["plot_locs"] == True:
        ts_dagger.set_data([fes.results["ts_loc"][i][1], fes.results["ts_loc"][i][0]])
        minima_list_x = [fes.results["minima_locs"][i][0][0],fes.results["minima_locs"][i][1][0]]
        minima_list_y = [fes.results["minima_locs"][i][0][1],fes.results["minima_locs"][i][1][1]]
        react_prod_dot.set_data([minima_list_x, minima_list_y])
    
        xpath = fes.results["mfep"][i][0][:,1][::options["skip_path_nodes"]]
        ypath = fes.results["mfep"][i][0][:,0][::options["skip_path_nodes"]]
        if options["skip_path_nodes"] != 1:
            xpath[-1] = fes.results["mfep"][i][0][-1,1]
            ypath[-1] = fes.results["mfep"][i][0][-1,0]
        path_trace.set_data(xpath, ypath)
    
        
    return annotation, fes_plot, react_prod_dot, ts_dagger, ts_line, ts_rev_line, dG_line, path_trace
    
    
    