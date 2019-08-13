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

mach = np.linspace(min(xt[:, 0]), max(xt[:, 0]), 50)
mach_step = mach[1] - mach[0]
alt = np.linspace(min(xt[:, 1]), max(xt[:, 1]), 50)
alt_step = alt[1] - alt[0]
throttle = np.linspace(min(xt[:, 2]), max(xt[:, 2]), 50)
throttle_step = throttle[1] - throttle[0]

x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)

source = ColumnDataSource(data=dict(x=x, y=y, z=np.random.rand(50, 50), alt=alt, mach=mach, throttle=throttle))

def make_predictions(x):
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

def contour_data():

    mach_value = mach_slider.value
    alt_value = alt_slider.value
    throttle_value = throttle_slider.value

    n = 50
    nx = 3
    ny = 2

    xe = np.zeros((n, n, nx))
    ye = np.zeros((n, n, ny))
    
    x0_list = [mach_value, alt_value, throttle_value]

    for ix in range(nx):
        xe[:,:, ix] = x0_list[ix]
    xlins = np.linspace(min(mach), max(mach), n)
    ylins = np.linspace(min(alt), max(alt), n)

    X, Y = np.meshgrid(xlins, ylins)
    xe[:,:, 0] = X
    xe[:,:, 1] = Y
    
    # print(xe)
    ye[:,:,:] = make_predictions(xe.reshape((n**2, nx))).reshape((n, n, ny))
    Z = ye[:,:,0].flatten()
    Z = Z.reshape(n, n)

    # Update data source
    source.data = dict(z=Z)

    ### Create Contour Plot

    contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
    contour_plot.x_range.range_padding = 0
    contour_plot.y_range.range_padding = 0
    contour_plot.plot_width = 500
    contour_plot.plot_height = 500
    contour_plot.xaxis.axis_label = "Mach"
    contour_plot.yaxis.axis_label = "Altitude"
    contour_plot.min_border_left = 100

    contour_plot.image(image=[source.data['z']], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")

    return contour_plot

def update(attr, old, new):
    layout.children[0] = contour_data()

mach_slider = Slider(start=min(mach), end=max(mach), value=0, step=mach_step, title="Mach")
mach_slider.on_change('value', update)

alt_slider = Slider(start=min(alt), end=max(alt), value=0, step=alt_step, title="Altitude")
alt_slider.on_change('value', update)

throttle_slider = Slider(start=min(throttle), end=max(throttle), value=0, step=throttle_step, title="Throttle") 
throttle_slider.on_change('value', update)


sliders = row(
    column(mach_slider, alt_slider, throttle_slider),
)
layout = row(contour_data(), sliders)

curdoc().add_root(layout)
curdoc().title = 'MultiView'













# def contour_data(attrname, old, new):
    
#     # Get the current slider values
#     mach_value = mach_slider.value
#     alt_value = alt_slider.value
#     throttle_value = throttle_slider.value

#     n = 50
#     nx = 3
#     ny = 2

#     xe = np.zeros((n, n, nx))
#     ye = np.zeros((n, n, ny))
    
#     x0_list = [mach_value, alt_value, throttle_value]

#     for ix in range(nx):
#         xe[:,:, ix] = x0_list[ix]
#     xlins = np.linspace(min(mach), max(mach), n)
#     ylins = np.linspace(min(alt), max(alt), n)

#     X, Y = np.meshgrid(xlins, ylins)
#     xe[:,:, 0] = X
#     xe[:,:, 1] = Y
    
#     # print(xe)
#     ye[:,:,:] = make_predictions(xe.reshape((n**2, nx))).reshape((n, n, ny))
#     Z = ye[:,:,0].flatten()
#     Z = Z.reshape(n, n)
    
#     source.data = dict(z=Z)
#     # print(source.data['z'])


#     contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
#     contour_plot.x_range.range_padding = 0
#     contour_plot.y_range.range_padding = 0
#     contour_plot.plot_width = 500
#     contour_plot.plot_height = 500
#     # contour_plot.source = 
#     contour_plot.xaxis.axis_label = "Mach"
#     contour_plot.yaxis.axis_label = "Altitude"
#     contour_plot.min_border_left = 100
#     # must give a vector of image data for image parameter
#     contour_plot.image(image=[source.data['z']], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")

#     return contour_plot


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

# def contour():

#     contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
#     contour_plot.x_range.range_padding = 0
#     contour_plot.y_range.range_padding = 0
#     contour_plot.plot_width = 500
#     contour_plot.plot_height = 500
#     # contour_plot.source = 
#     contour_plot.xaxis.axis_label = "Mach"
#     contour_plot.yaxis.axis_label = "Altitude"
#     contour_plot.min_border_left = 100
#     # must give a vector of image data for image parameter
#     contour_plot.image(image=[source.data['z']], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")

#     return contour_plot

# # contour_plot = figure(tooltips=[("Mach", "$x"), ("Altitude", "$y"), ("Thrust", "@image")])
# # contour_plot.x_range.range_padding = 0
# # contour_plot.y_range.range_padding = 0
# # contour_plot.plot_width = 500
# # contour_plot.plot_height = 500
# # # contour_plot.source = 
# # contour_plot.xaxis.axis_label = "Mach"
# # contour_plot.yaxis.axis_label = "Altitude"
# # contour_plot.min_border_left = 100
# # # must give a vector of image data for image parameter
# # contour_plot.image(image=[source.data['z']], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")


# for w in [mach_slider, alt_slider, throttle_slider]:
#     w.on_change('value', contour_data)
#     print(source.data['z'])

# # Layout for sliders
# layout = row(
#     column(mach_slider, alt_slider, throttle_slider),
# )
# # make a grid
# grid = gridplot([[contour_data(), layout]])

# # show the results
# curdoc().add_root(grid)



