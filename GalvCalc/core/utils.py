
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.surface_analysis import SurfaceEnergyPlotter
from pymatgen.util.plotting import pretty_plot


class MyPlotter(SurfaceEnergyPlotter):
    """
    A SurfaceEnergyPlotter subclass with a fixed chempot_vs_gamma method
    The main fix is that pretty_plot returns an Axes object while the code incorrectly called plt.gca()
    """

    def chempot_vs_gamma(
            self,
            ref_delu,
            chempot_range,
            miller_index=(),
            delu_dict=None,
            delu_default=0,
            JPERM2=False,
            show_unstable=False,
            ylim=None,
            plt=None,
            no_clean=False,
            no_doped=False,
            use_entry_labels=False,
            no_label=False,
    ):
        """
        Fixed chempot_vs_gamma method
        Main fix: correctly handle the Axes object returned by pretty_plot
        """
        if delu_dict is None:
            delu_dict = {}
        chempot_range = sorted(chempot_range)

        # fix: handle the plt parameter correctly
        if plt is None:
            # pretty_plot creates the figure and returns an Axes object
            ax = pretty_plot(width=8, height=7)
        else:
            # if a plt argument is passed, treat it as an Axes object
            ax = plt

        for hkl in self.all_slab_entries:
            if miller_index and hkl != tuple(miller_index):
                continue
            # Get the chempot range of each surface if we only
            # want to show the region where each slab is stable
            if not show_unstable:
                stable_u_range_dict = self.stable_u_range_dict(
                    chempot_range, ref_delu, no_doped=no_doped, delu_dict=delu_dict, miller_index=hkl
                )
            else:
                stable_u_range_dict = {}

            already_labelled = []
            label = ""
            for clean_entry in self.all_slab_entries[hkl]:
                urange = stable_u_range_dict[clean_entry] if not show_unstable else chempot_range
                # Don't plot if the slab is unstable, plot if it is.
                if urange != []:
                    label = clean_entry.label
                    if label in already_labelled:
                        label = None
                    else:
                        already_labelled.append(label)
                    if not no_clean:
                        if use_entry_labels:
                            label = clean_entry.label
                        if no_label:
                            label = ""
                        # fix: pass the correct Axes object
                        ax = self.chempot_vs_gamma_plot_one(
                            ax,  # pass the Axes object
                            clean_entry,
                            ref_delu,
                            urange,
                            delu_dict=delu_dict,
                            delu_default=delu_default,
                            label=label,
                            JPERM2=JPERM2,
                        )
                if not no_doped:
                    for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                        # Plot the adsorbed slabs
                        # Generate a label for the type of slab
                        urange = stable_u_range_dict[ads_entry] if not show_unstable else chempot_range
                        if urange != []:
                            if use_entry_labels:
                                label = ads_entry.label
                            if no_label:
                                label = ""
                            ax = self.chempot_vs_gamma_plot_one(
                                ax,  # pass the Axes object
                                ads_entry,
                                ref_delu,
                                urange,
                                delu_dict=delu_dict,
                                delu_default=delu_default,
                                label=label,
                                JPERM2=JPERM2,
                            )

        # Make the figure look nice
        if JPERM2:
            ax.set_ylabel(r"Surface energy (J/m$^{2}$)")
        else:
            ax.set_ylabel(r"Surface energy (eV/$\AA^{2}$)")

        # fix: pass the correct arguments to chempot_plot_addons
        return self.chempot_plot_addons(ax, chempot_range, str(ref_delu).split("_")[1], ylim=ylim)

    def chempot_plot_addons(self, ax, xrange, ref_el, pad=2.4, rect=None, ylim=None):
        """
        Fixed plot-decorating function that keeps the same signature as the parent class
        """
        # Make the figure look nice
        ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
        ax.set_xlabel(rf"Chemical potential $\Delta\mu_{{{ref_el}}}$ (eV)")

        ylim = ylim or ax.get_ylim()

        # rotate the x-axis labels
        ax.tick_params(axis='x', rotation=60)

        # set the axis limits
        ax.set_ylim(ylim)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim)

        # adjust the layout with plt.tight_layout, keeping the parent parameters
        plt.tight_layout(pad=pad, rect=rect or [-0.047, 0, 0.84, 1])

        # add reference lines
        ax.plot([xrange[0], xrange[0]], ylim, "--k")
        ax.plot([xrange[1], xrange[1]], ylim, "--k")

        # add annotations
        xy_rich = [np.mean([xrange[1]]), np.mean(ylim)]
        ax.annotate(f"{ref_el}-rich", xy=xy_rich, xytext=xy_rich, rotation=90, fontsize=17)

        xy_poor = [np.mean([xlim[0]]), np.mean(ylim)]
        ax.annotate(f"{ref_el}-poor", xy=xy_poor, xytext=xy_poor, rotation=90, fontsize=17)

        return ax

    def list_methods(self):
        """List all available plotting methods with brief descriptions"""
        methods_info = {
            'chempot_vs_gamma': 'Plot surface energy vs chemical potential',
            'area_frac_vs_chempot_plot': 'Plot Wulff shape area fraction vs chemical potential',
            'wulff_from_chempot': 'Construct Wulff shape at specific chemical potential',
            'monolayer_vs_BE': 'Plot binding energy vs monolayer coverage',
            'BE_vs_clean_SE': 'Plot binding energy vs clean surface energy',
            'surface_chempot_range_map': 'Plot surface chemical potential range map (binary systems)',
            'get_stable_entry_at_u': 'Get most stable surface at specific chemical potential',
            'stable_u_range_dict': 'Get stable chemical potential ranges for surfaces',
        }

        print("Available Plotting Methods:")
        print("=" * 50)

        for method_name, description in methods_info.items():
            if hasattr(self, method_name):
                print(f"{method_name:30} - {description}")