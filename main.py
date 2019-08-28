# Import 

# Bokeh Imports
from bokeh.io import output_file, show, curdoc, save
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis, Slider, ColumnDataSource, ContinuousColorMapper, ColorBar, FixedTicker, BasicTicker, LinearColorMapper, Range1d
from bokeh.events import ButtonClick
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.widgets import TextInput, Select
from openmdao.devtools.debug import profiling


# Misc Imports
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import math
from scipy.spatial import cKDTree
import itertools
from collections import OrderedDict
from utility_functions import stack_outputs


class UnstructuredMetaModelVisualization(object):
    
    def __init__(self, info):
        self.info = info

        self.prob = self.info['prob']
        
        # self.input_data = [i for i in self.prob.model.interp.options[str('train:' + i)]]

        self.input_list = [i[0] for i in self.prob.model.interp._surrogate_input_names]
        self.output_list = [i[0] for i in self.prob.model.interp._surrogate_output_names]

        self.n = self.info['resolution']

        self.x_input = Select(title="X Input:", value=[x for x in self.input_list][0], 
        options=[x for x in self.input_list])
        self.x_input.on_change('value', self.x_input_update)
        
        self.y_input = Select(title="Y Input:", value=[x for x in self.input_list][1], 
        options=[x for x in self.input_list])
        self.y_input.on_change('value', self.y_input_update)

        self.z_input = Select(title="Z Input:", value=[x for x in self.input_list][2], 
        options=[x for x in self.input_list])
        self.z_input.on_change('value', self.z_input_update)

        self.output_value = Select(title="Output:", value=[x for x in self.output_list][0], 
        options=[x for x in self.output_list])
        self.output_value.on_change('value', self.output_value_update)


        self.slider_dict = {}
        self.input_data_dict = OrderedDict()

        # Setup for sliders
        for title, values in self.info['input_names'].items():
            slider_spacing = np.linspace(min(values), max(values), self.n)
            self.input_data_dict[title] = slider_spacing
            slider_step = slider_spacing[1] - slider_spacing[0]
            self.slider_dict[title] = Slider(start=min(values), end=max(values), value=min(values), step=slider_step, title=str(title))        

        # Need to remove z_input
        for name, slider_object in self.slider_dict.items():
            if name == self.x_input.value:
                self.x_input_slider = slider_object
                self.x_input_slider.on_change('value', self.scatter_plots_update)
            elif name == self.y_input.value:
                self.y_input_slider = slider_object
                self.y_input_slider.on_change('value', self.scatter_plots_update)
            elif name == self.z_input.value:
                self.z_input_slider = slider_object
                self.z_input_slider.on_change('value', self.update)


        self.x = np.linspace(0, 100, self.n)
        self.y = np.linspace(0, 100, self.n)

        self.nx = len(self.input_list)
        self.ny = len(self.output_list)

        self.x_index = 0
        self.y_index = 1
        
        self.output_variable = self.info['output_names'].index(self.output_value.value)

        self.slider_source = ColumnDataSource(data=self.input_data_dict)
        self.bot_plot_source = ColumnDataSource(data=dict(bot_slice_x=np.repeat(0,self.n), bot_slice_y=np.repeat(0,self.n)))
        self.left_plot_source = ColumnDataSource(data=dict(left_slice_x=np.repeat(0,self.n), left_slice_y=np.repeat(0,self.n), 
        x1=np.repeat(0,self.n), x2=np.repeat(0,self.n)))
        self.source = ColumnDataSource(data=dict(x=self.x, y=self.y))

        self.scatter_distance = TextInput(value="0.1", title="Scatter Distance")
        self.dist_range = float(self.scatter_distance.value)
        self.scatter_distance.on_change('value', self.scatter_input)

        sliders = []
        for i in self.slider_dict.values():
            sliders.append(i)

        self.sliders = row(
            column(*sliders, self.x_input,
            self.y_input, self.z_input, self.output_value, self.scatter_distance)
        )
               
        self.layout = row(self.contour_data(), self.left_plot(), self.sliders)
        self.layout2 = row(self.bot_plot())
        curdoc().add_root(self.layout)
        curdoc().add_root(self.layout2)
        curdoc().title = 'MultiView'

    def make_predictions(self, data):

        outputs = {i : [] for i in self.output_list}
        print("Making Predictions")
        # options._dict to get data
        # _surrogate_input_names to get inputs
        # _surrogate_output_names to get outputs

        # Parse dict into [n**2, number of inputs] list
        inputs = np.empty([self.n**2, self.nx])
        for idx, values in enumerate(data.values()):
            inputs[:, idx] = values.flatten()

        # pair data points with their respective prob name. Loop to make predictions
        for idx, tup in enumerate(inputs):
            for name, val in zip(data.keys(), tup):
                self.prob['interp.'+ name] = val
            self.prob.run_model()
            for i in self.output_list:
                outputs[i].append(float(self.prob['interp.' + i]))

        return stack_outputs(outputs)

    def contour_data(self):

        n = self.n
        xe = np.zeros((n, n, self.nx))
        ye = np.zeros((n, n, self.ny))

        self.slider_value_and_name = OrderedDict()
        for title, info in self.slider_dict.items():
            self.slider_value_and_name[title] = info.value

        self.input_point_list = list(self.slider_value_and_name.values())

        for ix in range(self.nx):
            xe[:, :, ix] = self.input_point_list[ix]
    
        for title, values in self.input_data_dict.items():
            if title == self.x_input.value:
                xlins = values
                dw = max(values)
            if title == self.y_input.value:
                ylins = values
                dh = max(values)

        X, Y = np.meshgrid(xlins, ylins)
        # Figure out if x_index is always the 0 column or if it would change in higher dimensional spaces
        xe[:, :, self.x_index] = X
        xe[:, :, self.y_index] = Y

        # This block puts the x and y inputs first and then appends any other values afterwards 
        pred_dict = {}
        self.input_list = [self.x_input.value, self.y_input.value]
        for title in self.slider_value_and_name.keys():
            if title == self.x_input.value or title == self.y_input.value:
                pass
            else:
                self.input_list.append(title)
        
        for idx, title in enumerate(self.slider_value_and_name.keys()):
            pred_dict.update({title: xe[:, :, idx]})
        pred_dict_ordered = OrderedDict((k, pred_dict[k]) for k in self.input_list)
        print(pred_dict_ordered.keys())

        ye[:, :, :] = self.make_predictions(pred_dict_ordered).reshape((n, n, self.ny))
        Z = ye[:, :, self.output_variable]
        Z = Z.reshape(n, n)
        self.Z = Z

        try:
            self.source.add(Z, 'z')
        except KeyError:
            print("KeyError")

        # Color bar formatting
        color_mapper =  LinearColorMapper(palette="Viridis11", low=np.amin(Z), high=np.amax(Z))
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12, location=(0,0))

        # Contour Plot
        self.contour_plot = figure(tooltips=[(self.x_input.value, "$x"), (self.y_input.value, "$y"), (self.output_value.value, "@image")], tools="")
        self.contour_plot.x_range.range_padding = 0
        self.contour_plot.y_range.range_padding = 0
        self.contour_plot.plot_width = 600
        self.contour_plot.plot_height = 500
        self.contour_plot.xaxis.axis_label = self.x_input.value
        self.contour_plot.yaxis.axis_label = self.y_input.value
        self.contour_plot.min_border_left = 0
        self.contour_plot.add_layout(color_bar, 'right')
        self.contour_plot.x_range = Range1d(0, max(xlins))
        self.contour_plot.y_range = Range1d(0, max(ylins))

        self.contour_plot.image(image=[self.source.data['z']], x=0, y=0, dh=dh, dw=dw, palette="Viridis11")

        data = self.training_points()
        if len(data):
            data = np.array(data)
            self.contour_plot.circle(x=data[:, 0], y=data[:,1], size=5, color='white', alpha=0.25)

        return self.contour_plot        

        

    def left_plot(self):
        
        for title, values in self.input_data_dict.items():
            if title == self.x_input.value:
                self.x_value = self.x_input_slider.value
                mach_index = np.where(np.around(self.input_data_dict[title], 5) == np.around(self.x_value, 5))[0]

            elif title == self.y_input.value:
                self.y_data = self.input_data_dict[title]

        z_data = self.Z[:, mach_index].flatten()

        try:
            self.source.add(z_data, 'left_slice')
        except KeyError:
            self.source.data['left_slice'] = z_data

        x = self.source.data['left_slice']
        y = self.slider_source.data[self.y_input.value]

        s1 = figure(plot_width=200, plot_height=500, x_range=(min(x), max(x)), y_range=(0,max(self.y_data)), title="{} vs {}".format(self.y_input.value, self.output_value.value), tools="")
        s1.xaxis.axis_label = self.output_value.value
        s1.yaxis.axis_label = self.y_input.value
        s1.line(x, y)

        data = self.training_points()
        vert_color = np.zeros((len(data), 1))
        for i,info in enumerate(data):
            alpha = np.abs(info[0] - self.x_value) / self.limit_range[self.x_index]
            if alpha < self.dist_range:
                vert_color[i, -1] = (1 - alpha / self.dist_range) * info[-1]

        color = np.column_stack((data[:,-4:-1] - 1, vert_color))
        alphas = [0 if math.isnan(x) else x for x in color[:, 3]]
        s1.scatter(x=data[:, 3], y=data[:, 1], line_color=None, fill_color='#000000', fill_alpha=alphas)

        self.left_plot_source.data = dict(left_slice_x=np.repeat(self.x_value, self.n), left_slice_y=self.y_data, 
        x1=np.array([x+self.dist_range for x in np.repeat(self.x_value, self.n)]), x2=np.array([x-self.dist_range for x in np.repeat(self.x_value, self.n)]))

        # self.contour_plot.harea(y='left_slice_y', x1='x1', x2='x2', source=self.left_plot_source, color='gray', fill_alpha=0.25)
        # self.contour_plot.line('left_slice_x', 'left_slice_y', source=self.left_plot_source, color='black', line_width=2)
        
        return s1

    def bot_plot(self):

        for title, values in self.input_data_dict.items():
            if title == self.x_input.value:
                self.x_data = self.input_data_dict[title]
                
            elif title == self.y_input.value:
                self.y_value = self.y_input_slider.value
                y_range = [min(values), max(values)]
                alt_index = np.where(np.around(self.input_data_dict[title], 5) == np.around(self.y_value, 5))[0]
        
        z_data = self.Z[alt_index].flatten()

        try:
            self.source.add(z_data, 'bot_slice')
        except KeyError:
            self.source.data['bot_slice'] = z_data

        x = self.slider_source.data[self.x_input.value]
        y = self.source.data['bot_slice']

        s2 = figure(plot_width=550, plot_height=200, x_range=(0,max(self.x_data)), y_range=(min(y), max(y)), 
        title="{} vs {}".format(self.x_input.value, self.output_value.value), tools="")
        s2.xaxis.axis_label = self.x_input.value
        s2.yaxis.axis_label = self.output_value.value
        s2.line(x, y)

        data = self.training_points()
        horiz_color = np.zeros((len(data), 1))
        for i,info in enumerate(data):
            alpha = np.abs(info[1] - self.y_value) / self.limit_range[self.y_index]
            if alpha < self.dist_range:
                horiz_color[i, -1] = (1 - alpha / self.dist_range) * info[-1]
        
        color = np.column_stack((data[:,-4:-1] - 1, horiz_color))
        alphas = [0 if math.isnan(x) else x for x in color[:, 3]]
        s2.scatter(x=data[:, 0], y=data[:, 3], line_color=None, fill_color='#000000', fill_alpha=alphas)

        dist = self.dist_range * (y_range[1] - y_range[0])
        y1 = np.array([i+dist for i in np.repeat(self.y_value, self.n)])
        y2 = np.array([i-dist for i in np.repeat(self.y_value, self.n)])
        
        self.bot_plot_source.data = dict(bot_slice_x=self.x_data, bot_slice_y=np.repeat(self.y_value, self.n), y1=y1, y2=y2)
        # self.contour_plot.varea(x='bot_slice_x', y1='y1', y2='y2', source=self.bot_plot_source, color='gray', fill_alpha=0.25, name='varea')
        self.contour_plot.line('bot_slice_x', 'bot_slice_y', source=self.bot_plot_source, color='black', line_width=2)

        return s2

    def update_all_plots(self):
        self.layout.children[0] = self.contour_data()
        self.layout.children[1] = self.left_plot()
        self.layout2.children[0] = self.bot_plot()

    def update_subplots(self):
        self.layout.children[1] = self.left_plot()
        self.layout2.children[0] = self.bot_plot()

    # Callback functions
    def update(self, attr, old, new):
        self.update_all_plots()

    def scatter_plots_update(self, attr, old, new):
        
        self.update_subplots()

    def scatter_input(self, attr, old, new):
        self.dist_range = float(new)
        self.update_all_plots()

    def input_dropdown_checks(self,x,y,z):
        # Might be able to put all the false cases into the if statement "if x == y or x == z:" etc
        if x == y:
            return False
        elif x == z:
            return False
        elif y == z:
            return False
        else:
            return True
    
    def x_input_update(self, attr, old, new):
        if self.input_dropdown_checks(new, self.y_input.value, self.z_input.value) == False:
            # self.x_input.value = old
            raise ValueError("Inputs should not equal each other")
        else:
            self.update_all_plots()

    def y_input_update(self, attr, old, new):
        if self.input_dropdown_checks(self.x_input.value, new, self.z_input.value) == False:
            # self.y_input.value = old
            raise ValueError("Inputs should not equal each other")
        else: 
            self.update_all_plots()

    def z_input_update(self, attr, old, new):
        if self.input_dropdown_checks(self.x_input.value, self.y_input.value, new) == False:
            # self.z_input.value = old
            raise ValueError("Inputs should not equal each other")
        else:
            self.update_all_plots()

    def output_value_update(self, attr, old, new):
        self.output_variable = self.info['output_names'].index(new)
        self.update_all_plots()



    def remove_glyphs(self, figure, glyph_name_list):
        renderers = figure.select(dict(type=GlyphRenderer))
        for r in renderers:
            if r.name in glyph_name_list:
                col = r.glyph.y
                r.data_source.data[col] = [np.nan] * len(r.data_source.data[col])
        
    def remove_glyphs_x(self, figure, glyph_name_list):
        renderers = figure.select(dict(type=GlyphRenderer))
        for r in renderers:
            if r.name in glyph_name_list:
                col = r.glyph.x
                r.data_source.data[col] = [np.nan] * len(r.data_source.data[col])        

    def training_points(self):

        xt = self.info['scatter_points'][0] # Input Data
        yt = self.info['scatter_points'][1] # Output Data
        output_variable = self.info['output_names'].index(self.output_value.value)

        data = np.zeros((0, 8))
        limits = np.array(self.info['bounds'])
        self.limit_range = limits[:, 1] - limits[:, 0]

        infos = np.vstack((xt[:, self.x_index], xt[:, self.y_index])).transpose()
        points = xt.copy()
        points[:, self.x_index] = self.input_point_list[self.x_index]
        points[:, self.y_index] = self.input_point_list[self.y_index]
        points = np.divide(points, self.limit_range)
        tree = cKDTree(points)
        dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.input_point_list, self.limit_range)
        dists, idx = tree.query(scaled_x0, k=len(xt), distance_upper_bound=dist_limit)
        idx = idx[idx != len(xt)]
        data = np.zeros((len(idx), 8))

        for dist_index, i in enumerate(idx):
            if i != len(xt):
                info = np.ones((8))
                info[0:2] = infos[i, :]
                info[2] = dists[dist_index] / dist_limit
                info[3] = yt[i, output_variable]
                info[7] = (1. - info[2] / self.dist_range) ** 0.5
                data[dist_index] = info

        return data


# TODO: https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html check out bands
