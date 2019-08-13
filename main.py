# Import 

# Bokeh Imports
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis, Slider, ColumnDataSource

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

        self.x0_list = np.ones(self.info['num_of_inputs'])
        self.x_index = 0
        self.y_index = 1
        self.output_variable = self.info['output_variable']
        self.dist_range = self.info['dist_range']
        xt = self.info['scatter_points'][0]

        self.N = self.info['resolution']
        self.n = 50
        self.x = np.linspace(0, 100, self.N)
        self.y = np.linspace(0, 100, self.N)

        self.mach = np.linspace(min(xt[:, 0]), max(xt[:, 0]), self.n)
        self.mach_step = self.mach[1] - self.mach[0]
        self.alt = np.linspace(min(xt[:, 1]), max(xt[:, 1]), self.n)
        self.alt_step = self.alt[1] - self.alt[0]
        self.throttle = np.linspace(min(xt[:, 2]), max(xt[:, 2]), self.n)
        self.throttle_step = self.throttle[1] - self.throttle[0]

        self.mach_slider = Slider(start=min(self.mach), end=max(self.mach), value=0, step=self.mach_step, title="Mach")
        self.alt_slider = Slider(start=min(self.alt), end=max(self.alt), value=0, step=self.alt_step, title="Altitude")
        self.throttle_slider = Slider(start=min(self.throttle), end=max(self.throttle), value=0, step=self.throttle_step, title="Throttle") 

        self.source = ColumnDataSource(data=dict(x=self.x, y=self.y, alt=self.alt, mach=self.mach, throttle=self.throttle))

    def plot_contour(self):

        n = self.info['resolution']
        nx = self.info['num_of_inputs']
        ny = self.info['num_of_outputs']
        xt = self.info['scatter_points'][0]

        xe = np.zeros((n, n, nx))
        ye = np.zeros((n, n, ny))

        for ix in range(nx):
            xe[:,:, ix] = self.x0_list[ix]
        xlins = np.linspace(min(self.mach), max(self.mach), n)
        ylins = np.linspace(min(self.alt), max(self.alt), n)

        X, Y = np.meshgrid(xlins, ylins)
        xe[:,:, self.x_index] = X
        xe[:,:, self.y_index] = Y

        ye[:,:,:] = self.make_predictions(xe.reshape((n**2, nx))).reshape((n, n, ny))
        self.Z = ye[:,:,self.output_variable].flatten()

        self.Z = self.Z.reshape(n, n)
        
        self.source.add(self.Z, name='Z')



    def layout_plots(self):

        output_file("layout.html")

        mach = self.mach
        alt = self.alt
        throttle = self.throttle

         

        contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
        contour_plot.x_range.range_padding = 0
        contour_plot.y_range.range_padding = 0
        contour_plot.plot_width = 500
        contour_plot.plot_height = 500
        contour_plot.xaxis.axis_label = "Mach"
        contour_plot.yaxis.axis_label = "Altitude"
        contour_plot.min_border_left = 100
        # must give a vector of image data for image parameter
        contour_plot.image(image=[self.source.data['Z']], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")

        # Altitude vs Thrust
        try:
            s1 = figure(plot_width=200, plot_height=500, y_range=(0,max(alt)), title="Altitude vs Thrust")
            s1.xaxis.axis_label = "Thrust"
            s1.yaxis.axis_label = "Altitude"
            s1.line(self.source.data['left_slice'], self.source.data['alt'], source = self.source)
        except KeyError:
            print('Left slice is not part of source data')
            pass

        # Mach vs Thrust
        try:
            s3 = figure(plot_width=500, plot_height=200, x_range=(0,max(mach)), title='Mach vs Thrust')
            s3.xaxis.axis_label = "Mach"
            s3.yaxis.axis_label = "Thrust"
            s3.line(self.source.data['mach'], self.source.data['bot_slice'], source = self.source)
        except KeyError:
            print('Bottom slice is not a part of source data')

        layout = row(
            column(self.mach_slider, self.alt_slider, self.throttle_slider),
        )
        grid = gridplot([[s1, contour_plot, layout], [None, s3]])


        # show the results
        curdoc().add_root(grid)

    # def alt_vs_thrust_callback(self, attr, old, new):
    #     mach_index = np.where(np.around(self.mach, 5) == np.around(new, 5))[0]
    #     self.z_data = self.Z[mach_index].flatten()

    #     self.x0_list[0] = np.around(new, 5)
    #     print(new)
    #     self.plot_contour()
    #     # self.layout_plots()
    #     # self.source.add(self.z_data, name='left_slice')
    #     # self.layout_plots()


    # def mach_vs_thrust_callback(self, attr, old, new):
    #     alt_index = np.where(np.around(self.alt, 5) == np.around(new, 5))[0]
    #     z_data = self.Z[alt_index].flatten()

    #     self.source.add(z_data, name='bot_slice')

    def contour_callback(self, attr, old, new):

        
        pass

    def make_predictions(self, x):
        thrust = []
        tsfc = []
        print("Making Predictions")

        for i, j, k in x:
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

# info = {'num_of_inputs':3,
#         'num_of_outputs':2,
#         'resolution':50,
#         'dimension_names':[
#             'Mach number',
#             'Altitude, kft',
#             'Throttle'],
#         'X_dimension':0,
#         'Y_dimension':1,
#         'scatter_points':[xt, yt],
#         'output_variable': 0,
#         'dist_range': .1,
#         'output_names':[
#             'Thrust, 1e5 N',
#             'TSFC, 1/s']
# }


# viz = UnstructuredMetaModelVisualization(info)
# viz.plot_contour()
# viz.layout_plots()


# N is number of points we want to predict for
# n = 100

# mach = np.linspace(min(xt[:, 0]), max(xt[:, 0]), n)
# mach_step = mach[1] - mach[0]
# alt = np.linspace(min(xt[:, 1]), max(xt[:, 1]), n)
# alt_step = alt[1] - alt[0]
# throttle = np.linspace(min(xt[:, 2]), max(xt[:, 2]), n)
# throttle_step = throttle[1] - throttle[0]
# param = zip(mach, alt, throttle)

# def make_predictions(x):
#     thrust = []
#     tsfc = []
#     print("Making Predictions")

#     for i, j, k in x:
#         prob['interp.Mach'] = i
#         prob['interp.Alt'] = j
#         prob['interp.Throttle'] = k
#         prob.run_model()
#         thrust.append(float(prob['interp.Thrust']))
#         tsfc.append(float(prob['interp.TSFC']))

#     # Cast as np arrays to concatenate them together at the end
#     thrust = np.asarray(thrust)
#     tsfc = np.asarray(tsfc)

#     return np.stack([thrust, tsfc], axis=-1)


# nx = 3
# ny = 2
# x_index = 0
# y_index = 1
# output_variable = 0

# # Here we create a meshgrid so that we can take the X and Y
# # arrays and make pairs to send to make_predictions to get predictions
# xe = np.zeros((n, n, nx))
# ye = np.zeros((n, n, ny))

# # x0_list needs to take in values of the sliders
# # x0_list = np.ones(nx)
# x0_list = [0.38, 17.2, 0.5]
# for ix in range(nx):
#     xe[:,:, ix] = x0_list[ix]
# xlins = np.linspace(min(mach), max(mach), n)
# ylins = np.linspace(min(alt), max(alt), n)

# X, Y = np.meshgrid(xlins, ylins)
# xe[:,:, x_index] = X
# xe[:,:, y_index] = Y

# ye[:,:,:] = make_predictions(xe.reshape((n**2, nx))).reshape((n, n, ny))
# Z = ye[:,:,output_variable].flatten()

# Z = Z.reshape(n, n)


# #### Layout #####

# output_file("layout.html")

# ## Sample data points ##
# x = list(range(11))
# y0 = x
# y1 = [10 - i for i in x]
# y2 = [abs(i - 5) for i in x]


# N = 500
# x = np.linspace(0, 100, N)
# y = np.linspace(0, 100, N)

# def round_half_down(n, decimals=0):
#     multiplier = 10 ** decimals
#     return math.ceil(n*multiplier - 0.5) / multiplier

# source = ColumnDataSource(data=dict(x=x, y=y, z=Z, left_slice=Z, bot_slice=Z, alt=alt, mach=mach, throttle=throttle))



# def alt_vs_thrust_callback(attr, old, new):
#     mach_index = np.where(np.around(mach, 5) == np.around(new, 5))[0]
#     z_data = Z[mach_index].flatten()

#     source.data['left_slice'] = z_data

# def mach_vs_thrust_callback(attr, old, new):
#     alt_index = np.where(np.around(alt, 5) == np.around(new, 5))[0]
#     z_data = Z[alt_index].flatten()

#     source.data['bot_slice'] = z_data


# # Put units in later
# mach_slider = Slider(start=min(mach), end=max(mach), value=0, step=mach_step, title="Mach")
# alt_slider = Slider(start=min(alt), end=max(alt), value=0, step=alt_step, title="Altitude")
# throttle_slider = Slider(start=min(throttle), end=max(throttle), value=0, step=throttle_step, title="Throttle")

# mach_slider.on_change('value', alt_vs_thrust_callback)
# alt_slider.on_change('value', mach_vs_thrust_callback)

# contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
# contour_plot.x_range.range_padding = 0
# contour_plot.y_range.range_padding = 0
# contour_plot.plot_width = 500
# contour_plot.plot_height = 500
# contour_plot.xaxis.axis_label = "Mach"
# contour_plot.yaxis.axis_label = "Altitude"
# contour_plot.min_border_left = 100
# # must give a vector of image data for image parameter
# contour_plot.image(image=[Z], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")

# # create a new plot
# s1 = figure(plot_width=200, plot_height=500, y_range=(0,max(alt)), title="Altitude vs Thrust")
# s1.xaxis.axis_label = "Thrust"
# s1.yaxis.axis_label = "Altitude"
# s1.line('left_slice', 'alt', source = source)

# # create and another
# s3 = figure(plot_width=500, plot_height=200, x_range=(0,max(mach)), title='Mach vs Thrust')
# s3.xaxis.axis_label = "Mach"
# s3.yaxis.axis_label = "Thrust"
# s3.line('mach', 'bot_slice', source = source)

# # Layout for sliders
# layout = row(
#     column(mach_slider, alt_slider, throttle_slider),
# )
# # make a grid
# grid = gridplot([[s1, contour_plot, layout], [None, s3]])

# # show the results
# curdoc().add_root(grid)



