# Import 

# Bokeh Imports
from bokeh.io import output_file, show, curdoc, save
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis, Slider, ColumnDataSource, ContinuousColorMapper, ColorBar, FixedTicker, BasicTicker, LinearColorMapper, Button
from bokeh.events import ButtonClick
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.widgets import TextInput

# Misc Imports
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import math
from scipy.spatial import cKDTree
import itertools

# ======================================================================
#                       Engine Setup
# ======================================================================
# Load Data from John:
tmp = np.loadtxt('UHB.outputFLOPS')

# net thrust = gross thrust - ram drag (and convert to N)
tmp[:, 3] = (tmp[:, 3] - tmp[:, 4])*4.44822162

# Need to replace power code with fraction Tmax since there is not
# enough precision:
for i in range(len(tmp)):
    if tmp[i, 2] == 50:
        tmax = tmp[i, 3]
    tmp[i, 2] = tmp[i, 3] / tmax

# Converting units and identifying column titles
engineOptions = {'mach':tmp[:, 0],
           'altitude':tmp[:, 1]*.3048, # Alt in m
           'throttle':tmp[:, 2],
           'thrust':tmp[:, 3],
           'tsfc':tmp[:, 6]/3600.0, # TSFC in 1/s
           'rbfType':'cubic'}

# Creating empty arrays
nt = len(tmp)
xt = np.zeros((nt, 3))
yt = np.zeros((nt, 2))

# Mach in column 0 of xt
xt[:, 0] = tmp[:, 0]
# Altitude in column 1 of xt
xt[:, 1] = tmp[:, 1] / 1e3
# Trottle in column 2 of xt
xt[:, 2] = tmp[:, 2]

# Thrust in column 0 of yt
yt[:, 0] = tmp[:, 3] / 1e5
# tsfc in column 1 of yt
yt[:, 1] = tmp[:, 6] / 3600.

# Set the limits of x
xlimits = np.array([
    [0.0, 0.9],
    [0., 43.],
    [0, 1]
])

# Initial surrogate call
#### CHANGE THIS TO KRIGING SURROGATE WHEN GENERAL PLOTS ARE WORKING
interp = om.MetaModelUnStructuredComp(default_surrogate=om.ResponseSurface())
# Inputs
interp.add_input('Mach', 0., training_data=xt[:, 0])
interp.add_input('Alt', 0., training_data=xt[:, 1])
interp.add_input('Throttle', 0., training_data=xt[:, 2])

# Outputs
interp.add_output('Thrust', training_data=yt[:, 0])
interp.add_output('TSFC', training_data=yt[:, 1])

# Create the problem setup
prob = om.Problem()
prob.model.add_subsystem('interp', interp)
prob.setup()


class UnstructuredMetaModelVisualization(object):
    
    def __init__(self, info):
        self.info = info

        self.n = 50
        self.mach = np.linspace(min(xt[:, 0]), max(xt[:, 0]), self.n)
        self.mach_step = self.mach[1] - self.mach[0]
        self.alt = np.linspace(min(xt[:, 1]), max(xt[:, 1]), self.n)
        self.alt_step = self.alt[1] - self.alt[0]
        self.throttle = np.linspace(min(xt[:, 2]), max(xt[:, 2]), self.n)
        self.throttle_step = self.throttle[1] - self.throttle[0]

        self.x = np.linspace(0, 100, self.n)
        self.y = np.linspace(0, 100, self.n)

        self.nx = self.info['num_of_inputs']
        self.ny = self.info['num_of_outputs']

        self.x_index = 0
        self.y_index = 1
        

        self.output_variable = self.info['output_variable']

        self.mach_slider = Slider(start=min(self.mach), end=max(self.mach), value=0, step=self.mach_step, title="Mach")
        self.mach_slider.on_change('value', self.alt_vs_thrust_subplot_update)

        self.alt_slider = Slider(start=min(self.alt), end=max(self.alt), value=0, step=self.alt_step, title="Altitude")
        self.alt_slider.on_change('value', self.mach_vs_thrust_subplot_update)

        self.throttle_slider = Slider(start=min(self.throttle), end=max(self.throttle), value=0, step=self.throttle_step, title="Throttle") 
        self.throttle_slider.on_change('value', self.update)

        self.source = ColumnDataSource(data=dict(x=self.x, y=self.y))
        self.slider_source = ColumnDataSource(data=dict(alt=self.alt, mach=self.mach, throttle=self.throttle))

        self.scatter_distance = TextInput(value="0.1", title="Scatter Distance")
        self.dist_range = float(self.scatter_distance.value)
        self.scatter_distance.on_change('value', self.scatter_input)
        # self.button = Button(label="Foo", button_type="success")
        # self.button.on_event(ButtonClick, self.callback)

        self.sliders = row(
            column(self.mach_slider, self.alt_slider, self.throttle_slider, self.scatter_distance),
        )
        self.layout = row(self.contour_data(), self.alt_vs_thrust_subplot(), self.sliders)
        self.layout2 = row(self.mach_vs_thrust_subplot())
        curdoc().add_root(self.layout)
        curdoc().add_root(self.layout2)
        curdoc().title = 'MultiView'

    # def callback(self, event):
    #     save(self.layout)
    
    def make_predictions(self, data):
        thrust = []
        tsfc = []
        print("Making Predictions")

        for i, j, k in data:
            prob['interp.Mach'] = i
            prob['interp.Alt'] = j
            prob['interp.Throttle'] = k
            prob.run_model()
            thrust.append(float(prob['interp.Thrust']))
            tsfc.append(float(prob['interp.TSFC']))

        # Cast as np arrays to concatenate them together at the end
        thrust = np.asarray(thrust)
        tsfc = np.asarray(tsfc)

        return np.stack([thrust, tsfc], axis=-1)

    def contour_data(self):

        mach_value = self.mach_slider.value
        alt_value = self.alt_slider.value
        throttle_value = self.throttle_slider.value

        n = self.n
        xe = np.zeros((n, n, self.nx))
        ye = np.zeros((n, n, self.ny))
        self.x0_list = [mach_value, alt_value, throttle_value]

        for ix in range(self.nx):
            xe[:, :, ix] = self.x0_list[ix]
        xlins = np.linspace(min(self.mach), max(self.mach), n)
        ylins = np.linspace(min(self.alt), max(self.alt), n)

        X, Y = np.meshgrid(xlins, ylins)
        xe[:, :, self.x_index] = X
        xe[:, :, self.y_index] = Y

        ye[:, :, :] = self.make_predictions(xe.reshape((n**2, self.nx))).reshape((n, n, self.ny))
        Z = ye[:, :, self.output_variable]
        Z = Z.reshape(n, n)
        self.Z = Z

        try:
            self.source.add(Z, 'z')
        except KeyError:
            print("KeyError")

        # Color bar formatting
        # color_mapper = ContinuousColorMapper(palette="Viridis11", low=np.amin(Z), high=np.amax(Z))
        color_mapper =  LinearColorMapper(palette="Viridis11", low=np.amin(Z), high=np.amax(Z))
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12, location=(0,0))

        self.contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
        self.contour_plot.x_range.range_padding = 0
        self.contour_plot.y_range.range_padding = 0
        self.contour_plot.plot_width = 600
        self.contour_plot.plot_height = 500
        self.contour_plot.xaxis.axis_label = "Mach"
        self.contour_plot.yaxis.axis_label = "Altitude"
        self.contour_plot.min_border_left = 0
        self.contour_plot.add_layout(color_bar, 'right')

        self.contour_plot.image(image=[self.source.data['z']], x=0, y=0, dw=max(self.mach), dh=max(self.alt), palette="Viridis11")
        # contour_plot.line(x=mach_value, y=self.alt, color='black', source=)
        # self.contour_plot.line(x=0.4, y=[0.0, 43.0], color='black')

        data = self.training_points()
        if len(data):
            data = np.array(data)
            self.contour_plot.circle(x=data[:, 0], y=data[:,1], size=5, color='white', alpha=0.25)

        return self.contour_plot        

        

    def alt_vs_thrust_subplot(self):

        mach_value = self.mach_slider.value

        mach_index = np.where(np.around(self.mach, 5) == np.around(mach_value, 5))[0]
        z_data = self.Z[:, mach_index].flatten()

        try:
            self.source.add(z_data, 'left_slice')
        except KeyError:
            self.source.data['left_slice'] = z_data
        
        s1 = figure(plot_width=200, plot_height=500, y_range=(0,max(self.alt)), title="Altitude vs Thrust")
        s1.xaxis.axis_label = "Thrust"
        s1.yaxis.axis_label = "Altitude"
        s1.line(self.source.data['left_slice'], self.slider_source.data['alt'])

        data = self.training_points()
        vert_color = np.zeros((len(data), 1))
        for i,info in enumerate(data):
            alpha = np.abs(info[0] - mach_value) / self.limit_range[self.x_index]
            if alpha < self.dist_range:
                vert_color[i, -1] = (1 - alpha / self.dist_range) * info[-1]

        print("Dist range: ", self.dist_range)
        color = np.column_stack((data[:,-4:-1] - 1, vert_color))
        alphas = [0 if math.isnan(x) else x for x in color[:, 3]]
        s1.scatter(x=data[:, 3], y=data[:, 1], line_color=None, fill_color='#000000', fill_alpha=alphas)

        self.remove_glyphs(self.contour_plot, ['mach_line'])
        self.contour_plot.line(x=mach_value, y=self.alt, color='black', name='mach_line')
        
        return s1


    def mach_vs_thrust_subplot(self):

        # print(self.source.data['z'])
        alt_value = self.alt_slider.value
        

        alt_index = np.where(np.around(self.alt, 5) == np.around(alt_value, 5))[0]
        z_data = self.Z[alt_index].flatten()

        try:
            self.source.add(z_data, 'bot_slice')
        except KeyError:
            self.source.data['bot_slice'] = z_data
            # print("KeyError")

        s2 = figure(plot_width=550, plot_height=200, x_range=(0,max(self.mach)), title="Mach vs Thrust")
        s2.xaxis.axis_label = "Mach"
        s2.yaxis.axis_label = "Thrust"
        s2.line(self.slider_source.data['mach'], self.source.data['bot_slice'])
        

        data = self.training_points()
        horiz_color = np.zeros((len(data), 1))
        for i,info in enumerate(data):
            alpha = np.abs(info[1] - alt_value) / self.limit_range[self.y_index]
            if alpha < self.dist_range:
                horiz_color[i, -1] = (1 - alpha / self.dist_range) * info[-1]
        
        color = np.column_stack((data[:,-4:-1] - 1, horiz_color))
        alphas = [0 if math.isnan(x) else x for x in color[:, 3]]
        s2.scatter(x=data[:, 0], y=data[:, 3], line_color=None, fill_color='#000000', fill_alpha=alphas)
        
        self.remove_glyphs_x(self.contour_plot, ['alt_line'])
        self.contour_plot.line(x=self.mach, y=alt_value, color='black', name='alt_line')

        return s2

    def update(self, attr, old, new):
        self.layout.children[0] = self.contour_data()
        self.layout.children[1] = self.alt_vs_thrust_subplot()
        self.layout2.children[0] = self.mach_vs_thrust_subplot()

    def alt_vs_thrust_subplot_update(self, attr, old, new):
        self.layout.children[1] = self.alt_vs_thrust_subplot()

    def mach_vs_thrust_subplot_update(self, attr, old, new):
        self.layout2.children[0] = self.mach_vs_thrust_subplot()

    def scatter_input(self, attr, old, new):
        self.dist_range = float(new)

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
        xt = self.info['scatter_points'][0]
        yt = self.info['scatter_points'][1]

        data = np.zeros((0, 8))
        limits = np.array(self.info['bounds'])
        self.limit_range = limits[:, 1] - limits[:, 0]

        infos = np.vstack((xt[:, self.x_index], xt[:, self.y_index])).transpose()
        points = xt.copy()
        points[:, self.x_index] = self.x0_list[self.x_index]
        points[:, self.y_index] = self.x0_list[self.y_index]
        points = np.divide(points, self.limit_range)
        tree = cKDTree(points)
        dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.x0_list, self.limit_range)
        dists, idx = tree.query(scaled_x0, k=len(xt), distance_upper_bound=dist_limit)
        idx = idx[idx != len(xt)]
        data = np.zeros((len(idx), 8))

        for dist_index, i in enumerate(idx):
            if i != len(xt):
                info = np.ones((8))
                info[0:2] = infos[i, :]
                info[2] = dists[dist_index] / dist_limit
                info[3] = yt[i, self.output_variable]
                info[7] = (1. - info[2] / self.dist_range) ** 0.5
                data[dist_index] = info

        return data


# TODO: https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html check out bands
