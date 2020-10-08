# -*- coding: utf-8 -*-
"""
################
Highcharts Tools
################

*Created on Wed Jul 29 08:39:28 2015 by Axel Pahl*

Create Highcharts plots from dictionaries, molecule dictionaries or Pandas dataframes.
The latter two support structure tooltips.
"""

# 1. stdlib imports
import time
import string
import json
import colorsys
import os.path as op
import random

# 2. third-party imports
try:
    from misc_tools import apl_tools as apt
    AP_TOOLS = True
except ImportError:
    AP_TOOLS = False

# 3. internal project imports

# 4. IPython imports
# from IPython.html import widgets
from IPython.display import HTML, display

if op.isfile("lib/highcharts.js"):
    HC_LOCATION = "lib"
else:
    print("- no local installation of highcharts found, using web version.")
    HC_LOCATION = "http://code.highcharts.com"


# <script src="{hc_loc}/modules/heatmap.js"></script>
HIGHCHARTS = """
<script src="{hc_loc}/highcharts.js"></script>
<script src="{hc_loc}/highcharts-more.js"></script>
<script src="{hc_loc}/modules/heatmap.js"></script>
<script src="{hc_loc}/modules/exporting.js"></script>
""".format(hc_loc=HC_LOCATION)

CHART_TEMPL = """<div id="container_${id}" style="height: ${height}px"></div>
<script>
$$(function () {
    $$('#container_${id}').highcharts(

$chart

    );
});
</script>
"""

#: Currently supported chart kinds
CHART_KINDS = ["scatter", "column"]

#: Currently supported tooltip options
TOOLTIP_OPTIONS = "struct"

if AP_TOOLS:
    #: Library version
    VERSION = apt.get_commit(__file__)
    # I use this to keep track of the library versions I use in my project notebooks
    print("{:45s} (commit: {})".format(__name__, VERSION))
else:
    print("- loading highcharts...")

display(HTML(HIGHCHARTS))


class ColorScale():
    """Used for continuous coloring."""

    def __init__(self, num_values, val_min, val_max):
        self.num_values = num_values
        self.num_val_1 = num_values - 1
        self.value_min = val_min
        self.value_max = val_max
        self.value_range = self.value_max - self.value_min
        self.color_scale = []
        hsv_tuples = [(0.35 + ((x * 0.65) / (self.num_val_1)), 0.9, 0.9) for x in range(self.num_values)]
        rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
        for rgb in rgb_tuples:
            rgb_int = [int(255 * x) for x in rgb]
            self.color_scale.append('#{:02x}{:02x}{:02x}'.format(*rgb_int))

    def __call__(self, value, reverse=False):
        """return the color from the scale corresponding to the place in the value_min ..  value_max range"""
        pos = int(((value - self.value_min) / self.value_range) * self.num_val_1)

        if reverse:
            pos = self.num_val_1 - pos

        return self.color_scale[pos]


class Chart():
    """Available Chart kinds: scatter, column.

    Parameters:
        radius (int): Size of the points. Alias: *r*
        y_title (str): Used in the column plot as title of the y axis, default: ""."""

    def __init__(self, kind="scatter", **kwargs):
        if kind not in CHART_KINDS:
            raise ValueError("{} is not a supported chart kind ({})".format(kind, CHART_KINDS))

        self.kind = kind
        self.height = kwargs.get("height", 450)
        radius = kwargs.get("r", kwargs.get("radius", 5))  # accept "r" or "radius" for this option
        self.legend = kwargs.get("legend", None)
        self.y_title = kwargs.get("y_title", "")
        self.chart_id = time.strftime("%y%m%d%H%M%S")
        self.chart = {}
        self.chart["title"] = {"text": kwargs.get("title", "{} plot".format(self.kind))}
        self.chart["subtitle"] = {"text": kwargs.get("subtitle")}
        self.chart["series"] = []
        self.chart["plotOptions"] = {"scatter": {"marker": {"radius": radius}}}
        self.chart["credits"] = {'enabled': False}
        self.chart["yAxis"] = {"title": {"enabled": True, "text": self.y_title}}


    def _structure_tooltip(self, i):
        tooltip = []
        if self.arg_pid:
            tooltip.extend([str(self.dpid[i]), "<br>"])
        tooltip.extend(['<div style="width: 200px; height: 200px;">',
                       str(self.dmol[i]), "</div>"])
        return "".join(tooltip)


    def _extended_tooltip(self):
        ext_tt = [[] for idx in self.dx]
        for idx, _ in enumerate(self.dx):
            for field in self.arg_include_in_tooltip:
                ext_tt[idx].append("<b>{}</b>: {}".format(field, self.include[field][idx]))
            if self.arg_pid:
                ext_tt[idx].append("<b>{}</b>: {}".format(self.arg_pid, self.dpid[idx]))
        self.dpid = ["<br>".join(i) for i in ext_tt]
        self.arg_pid = True


    def _data_columns(self):
        """Generate the data for the Column plot"""
        data = []
        cats = []

        for i in range(self.dlen):
            try:
                cv = str(self.dx[i])
                dv = float(self.dy[i])
            except TypeError:
                continue

            cats.append(cv)
            data.append(dv)

        self.chart["series"].append({"name": self.arg_y, "data": data})
        self.chart["xAxis"]["categories"] = cats


    def _data_tuples(self, d):
        """Generate the data tuples required for Highcharts scatter plot."""
        data = []
        dx = d["x"]
        dy = d["y"]
        if self.arg_z:
            dz = d["z"]
        if self.arg_pid or self.arg_struct or self.arg_include_in_tooltip:
            dpid = d["id"]
        if self.arg_color_by:
            dcolorval = d["color_by"]

        for i in range(len(dx)):
            try:
                tmp_d = {"x": float(dx[i]), "y": float(dy[i])}
                if self.arg_z:
                    tmp_d["z"] = float(dz[i])
                if self.arg_pid or self.arg_struct or self.arg_include_in_tooltip:
                    tmp_d["id"] = str(dpid[i])
                if self.arg_color_by:
                    color_val = float(dcolorval[i])
                    color_code = self.color_scale(color_val, reverse=self.arg_reverse)
                    tmp_d["z"] = color_val
                    tmp_d["color"] = color_code
                    marker = {"fillColor": color_code,
                              "states": {"hover": {"fillColor": color_code}}}
                    tmp_d["marker"] = marker

                data.append(tmp_d)

            except TypeError:
                pass

        return data


    def _series_discrete(self):
        # [{"name": "A", "data": [{"x": 1, "y": 2}, {"x": 2, "y": 3}]},
        #  {"name": "B", "data": [{"x": 2, "y": 3}, {"x": 3, "y": 4}]}]
        self.arg_z = None  # not implemented yet
        series = []

        names = set(str(c) for c in self.dseries_by)
        data_series_x = {name: [] for name in names}
        data_series_y = {name: [] for name in names}
        if self.arg_pid:
            data_series_id = {name: [] for name in names}
        if self.arg_struct:
            data_series_mol = {name: [] for name in names}
        if self.arg_color_by:
            data_series_color = {name: [] for name in names}

        for i in range(self.dlen):
            series_by_str = str(self.dseries_by[i])
            data_series_x[series_by_str].append(float(self.dx[i]))
            data_series_y[series_by_str].append(float(self.dy[i]))
            if self.arg_struct:
                data_series_mol[series_by_str].append(self._structure_tooltip(i))
            elif self.arg_pid:
                data_series_id[series_by_str].append(str(self.dpid[i]))
            if self.arg_color_by:
                data_series_color[series_by_str].append(self.dcolor_by[i])

        for name in names:
            tmp_d = {"x": data_series_x[name], "y": data_series_y[name]}
            if self.arg_struct:
                tmp_d["id"] = data_series_mol[name]
            elif self.arg_pid:
                tmp_d["id"] = data_series_id[name]
            if self.arg_color_by:
                tmp_d["color_by"] = data_series_color[name]

            series_dict = {"name": name}
            series_dict["data"] = self._data_tuples(tmp_d)
            series.append(series_dict)

        return series


    def add_data(self, d, x="x", y="y", z=None, **kwargs):
        """Add the data to the chart.

        Parameters:
            d (dictionary or dataframe): The input dictionary
            str x , y [, z]: The keys for the properties to plot.

        Other Parameters:
            pid (str): The name of a (compound) id to be displayed in the tooltip.
                Defaults to *None*.
            tooltip (str): enable structure tooltips (currently only implemented for RDKit dataframes).
                Possible values: *"", "struct"*. Defaults to "".
            mol_col (str): Structure column in the df used for the tooltip
                (used if tooltip="struct"). Defaults to *"mol"*.
            color_by (str, None): property to use for coloring. Defaults to *None*
            series_by (str, None): property to use as series. Defaults to *None*
            color_mode (str): Point coloring mode. Alias: *mode*
                Available values: *"disc", "discrete", "cont", "continuos"*. Defaults to *"disc"*
            reverse (bool): Reverse the ColorScale. Defaults to *False*.
        """

        if x not in d or y not in d:
            raise KeyError("'{x}' and '{y}' are required parameters for scatter plot, but could not all be found in dict.".format(x=x, y=y))

        if len(d[x]) != len(d[y]):
            raise ValueError("'{x}' and '{y}' must have the same length.".format(x=self.arg_x, y=self.arg_y))

        self.arg_x = x
        self.arg_y = y
        self.arg_z = z
        self.arg_series_by = kwargs.get("series_by", None)
        self.arg_color_by = kwargs.get("color_by", None)
        self.arg_pid = kwargs.get("pid", None)
        self.arg_color_discrete = "disc" in kwargs.get("color_mode", kwargs.get("mode", "discrete"))
        self.arg_reverse = kwargs.get("reverse", False)
        self.arg_include_in_tooltip = kwargs.get("include_in_tooltip", kwargs.get("include", "xxx"))
        if not isinstance(self.arg_include_in_tooltip, list):
            self.arg_include_in_tooltip = [self.arg_include_in_tooltip]
        self.arg_jitter = kwargs.get("jitter", None)
        self.jitter_mag = kwargs.get("mag", 0.2)



        self.dx = list(d[x])
        self.dy = list(d[y])
        self.dlen = len(self.dx)

        if self.arg_jitter:
            for j in self.arg_jitter:
                if j == x:
                    for i, val in enumerate(self.dx):
                        self.dx[i] = val + self.jitter_mag * random.random() * 2 - self.jitter_mag

                if j == y:
                    for i, val in enumerate(self.dy):
                        self.dy[i] = val + self.jitter_mag * random.random() * 2 - self.jitter_mag

        if self.arg_pid:
            # pandas data series and pid == index
            if not isinstance(d[x], list) and self.arg_pid == d.index.name:
                self.dpid = list(d.index)
            else:
                # self.dpid = ["<b>{}:</b> {}".format(self.arg_pid, i) for i in list(d[self.arg_pid])]
                self.dpid = list(d[self.arg_pid])

            if self.dlen != len(self.dpid):
                raise ValueError("'{x}' and '{pid}' must have the same length.".format(x=self.arg_x, pid=self.arg_pid))
        else:
            self.arg_pid = None

        self.chart["xAxis"] = {"title": {"enabled": True, "text": self.arg_x}}

        #########################
        # plot-specific options #
        #########################
        if self.kind in ["scatter"]:
            if not self.y_title:
                self.chart["yAxis"] = {"title": {"enabled": True, "text": self.arg_y}}
            self.arg_tooltip = kwargs.get("tooltip", "")
            if self.arg_tooltip not in TOOLTIP_OPTIONS:
                print("- unknown tooltip option {}, setting to empty.".format(self.arg_tooltip))
                self.arg_tooltip = ""
            self.arg_struct = "struct" in self.arg_tooltip
            self.arg_mol_col = kwargs.get("mol_col", "mol")
            if self.arg_struct:
                self.dmol = list(d[self.arg_mol_col])

            self.include = {}
            includes = self.arg_include_in_tooltip[:]
            for field in includes:
                if field in d:
                    self.include[field] = list(d[field])
                else:
                    self.arg_include_in_tooltip.remove(field)  # remove fields that are not present in the data set

            if self.arg_pid or self.arg_include_in_tooltip:
                self._extended_tooltip()


        if self.kind == "scatter":
            self.chart["chart"] = {"type": "scatter", "zoomType": "xy"}
            # defining the tooltip
            self.chart["tooltip"] = {"useHTML": True}
            # self.chart["tooltip"]["headerFormat"] = "{y} vs. {x}<br>".format(x=x, y=y)
            self.chart["tooltip"]["headerFormat"] = ""

            point_format = ["<b>{x}:</b> {{point.x}}<br><b>{y}:</b> {{point.y}}".format(x=self.arg_x, y=self.arg_y)]
            if self.arg_color_by:
                point_format.append("<b>{color_by}:</b> {{point.z}}".format(color_by=self.arg_color_by))
            if self.arg_pid or self.arg_struct or self.arg_include_in_tooltip:
                point_format.append("{point.id}")
            self.chart["tooltip"]["pointFormat"] = "<br>".join(point_format)

            if not self.legend:
                self.chart["legend"] = {'enabled': False}
            else:
                self.chart["legend"] = {'enabled': True, "align": "right"}


            ############################
            # defining the data series #
            ############################
            if self.arg_series_by:
                if self.dlen != len(d[self.arg_series_by]):
                    raise ValueError("'{x}' and '{series_by}' must have the same length.".format(x=self.arg_x, series_by=self.arg_series_by))
                self.dseries_by = list(d[self.arg_series_by])
            if self.arg_color_by:
                if self.dlen != len(d[self.arg_color_by]):
                    raise ValueError("'{x}' and '{color_by}' must have the same length.".format(x=self.arg_x, color_by=self.arg_color_by))
                self.dcolor_by = list(d[self.arg_color_by])
                # self.chart["colorAxis"] = {"minColor": "#FFFFFF", "maxColor": "Highcharts.getOptions().colors[0]"}
                min_color_by = min(self.dcolor_by)
                max_color_by = max(self.dcolor_by)
                self.color_scale = ColorScale(20, min_color_by, max_color_by)
                # self.chart["colorAxis"] = {"min": min_color_by, "max": max_color_by,
                #                          "minColor": '#16E52B', "maxColor": '#E51616'}
                # if self.legend != False:
                #     self.chart["legend"] = {'enabled': True}
                if not self.chart["subtitle"]["text"]:
                    self.chart["subtitle"]["text"] = "colored by {} ({:.2f} .. {:.2f})".format(self.arg_color_by, min_color_by, max_color_by)
            if self.arg_z:
                if self.dlen != len(d[z]):
                    raise ValueError("'{x}' and '{z}' must have the same length.".format(x=self.arg_x, pid=self.arg_pid))
                self.dz = list(d[z])


            if self.arg_series_by:
                if self.legend is not False:
                    self.chart["legend"] = {'enabled': True, "title": {"text": self.arg_series_by},
                                            "align": "right"}
                    self.chart["tooltip"]["headerFormat"] = '<b>{series_by}: {{series.name}}</b><br>'.format(series_by=self.arg_series_by)
                series = self._series_discrete()
                self.chart["series"].extend(series)
            else:
                tmp_d = {"x": self.dx, "y": self.dy}
                if self.arg_struct:
                    tmp_d["id"] = [self._structure_tooltip(i) for i in range(self.dlen)]
                elif self.arg_pid:
                    tmp_d["id"] = self.dpid
                if self.arg_color_by:  # continuous values
                    tmp_d["color_by"] = self.dcolor_by

                data = self._data_tuples(tmp_d)
                self.chart["series"].append({"name": "series", "data": data})

        if self.kind == "column":
            self.chart["chart"] = {"type": "column", "zoomType": "xy"}
            self._data_columns()


    def show(self, debug=False):
        """Show the plot."""
        formatter = string.Template(CHART_TEMPL)
        # if debug:
        #     print(self.chart)
        chart_json = json.dumps(self.chart)
        html = formatter.substitute({"id": self.chart_id, "chart": chart_json,
                                     "height": self.height})
        # html = html.replace('"Highcharts.getOptions().colors[0]"', 'Highcharts.getOptions().colors[0]')
        if debug:
            print(self.dpid)
            print(html)
        return HTML(html)


def guess_id_prop(prop_list):  # try to guess an id_prop
    for prop in prop_list:
        if prop.lower().endswith("id"):
            return prop
    return None


# Quick Predefined Plots
def cpd_scatter(df, x, y, r=7, pid=None, **kwargs):
    """Predefined Plot #1.
    Quickly plot an RDKit Pandas dataframe or a molecule dictionary with structure tooltips."""

    if not pid:
        if isinstance(df, dict):
            prop_list = df.keys()
        else:
            prop_list = [df.index.name]
            prop_list.extend(df.columns.values)

        pid = guess_id_prop(prop_list)

    title = kwargs.get("title", "Compound Scatter Plot")
    scatter = Chart(title=title, r=r)
    scatter.add_data(df, x, y, pid=pid, **kwargs)
    return scatter.show()


# interactive exploration of an RDKit Pandas dataframe
def inspect_df(df, pid="Compound_Id", tooltip="struct"):
    """Use IPythons interactive widgets to visually and interactively explore an RDKit Pandas dataframe.
    TODO: implement!"""
    pass
