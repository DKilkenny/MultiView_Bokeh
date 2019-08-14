# Import 

# Bokeh Imports
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis, Slider, ColumnDataSource, ContinuousColorMapper, ColorBar, FixedTicker, BasicTicker, LinearColorMapper

# Misc Imports
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import math


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

        self.source = ColumnDataSource(data=dict(x=self.x, y=self.y))
        self.slider_source = ColumnDataSource(data=dict(alt=self.alt, mach=self.mach, throttle=self.throttle))

        self.mach_slider = Slider(start=min(self.mach), end=max(self.mach), value=0, step=self.mach_step, title="Mach")
        self.mach_slider.on_change('value', self.alt_vs_thrust_subplot_update)

        self.alt_slider = Slider(start=min(self.alt), end=max(self.alt), value=0, step=self.alt_step, title="Altitude")
        self.alt_slider.on_change('value', self.mach_vs_thrust_subplot_update)

        self.throttle_slider = Slider(start=min(self.throttle), end=max(self.throttle), value=0, step=self.throttle_step, title="Throttle") 
        self.throttle_slider.on_change('value', self.update)

        self.sliders = row(
            column(self.mach_slider, self.alt_slider, self.throttle_slider),
        )
        self.layout = row(self.contour_data(), self.alt_vs_thrust_subplot(), self.sliders)
        self.layout2 = row(self.mach_vs_thrust_subplot())
        curdoc().add_root(self.layout)
        curdoc().add_root(self.layout2)
        curdoc().title = 'MultiView'
    
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
        x0_list = [mach_value, alt_value, throttle_value]

        for ix in range(self.nx):
            xe[:, :, ix] = x0_list[ix]
        xlins = np.linspace(min(self.mach), max(self.mach), n)
        ylins = np.linspace(min(self.alt), max(self.alt), n)

        X, Y = np.meshgrid(xlins, ylins)
        xe[:, :, self.x_index] = X
        xe[:, :, self.y_index] = Y

        ye[:, :, :] = self.make_predictions(xe.reshape((n**2, self.nx))).reshape((n, n, self.ny))
        Z = ye[:, :, self.output_variable]
        Z = Z.reshape(n, n)
        self.Z = Z

        # print(self.source.data['z'])
        try:
            self.source.add(Z, 'z')
        except KeyError:
            print("KeyError")

        # Color bar formatting
        # color_mapper = ContinuousColorMapper(palette="Viridis11", low=np.amin(Z), high=np.amax(Z))
        color_mapper =  LinearColorMapper(palette="Viridis11", low=np.amin(Z), high=np.amax(Z))
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12, location=(0,0))

        contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
        contour_plot.x_range.range_padding = 0
        contour_plot.y_range.range_padding = 0
        contour_plot.plot_width = 600
        contour_plot.plot_height = 500
        contour_plot.xaxis.axis_label = "Mach"
        contour_plot.yaxis.axis_label = "Altitude"
        contour_plot.min_border_left = 0
        contour_plot.add_layout(color_bar, 'right')

        contour_plot.image(image=[self.source.data['z']], x=0, y=0, dw=max(self.mach), dh=max(self.alt), palette="Viridis11")

        return contour_plot

    def alt_vs_thrust_subplot(self):

        # print(self.source.data['z'])
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
        
        return s2

    def update(self, attr, old, new):
        self.layout.children[0] = self.contour_data()
        self.layout.children[1] = self.alt_vs_thrust_subplot()
        self.layout2.children[0] = self.mach_vs_thrust_subplot()

    def alt_vs_thrust_subplot_update(self, attr, old, new):
        self.layout.children[1] = self.alt_vs_thrust_subplot()

    def mach_vs_thrust_subplot_update(self, attr, old, new):
        self.layout2.children[0] = self.mach_vs_thrust_subplot()


