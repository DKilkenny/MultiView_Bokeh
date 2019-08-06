# Import 

# Bokeh Imports
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis

# Misc Imports
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt


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

# N is number of points we want to predict for
n = 100

mach = np.linspace(min(xt[:, 0]), max(xt[:, 0]), n)
alt = np.linspace(min(xt[:, 1]), max(xt[:, 1]), n)
throttle = np.linspace(min(xt[:, 2]), max(xt[:, 2]), n)
param = zip(mach, alt, throttle)

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


nx = 3
ny = 2
x_index = 0
y_index = 1
output_variable = 0

# Here we create a meshgrid so that we can take the X and Y
# arrays and make pairs to send to make_predictions to get predictions
xe = np.zeros((n, n, nx))
ye = np.zeros((n, n, ny))

x0_list = np.ones(nx)
for ix in range(nx):
    xe[:,:, ix] = x0_list[ix]
xlins = np.linspace(min(mach), max(mach), n)
ylins = np.linspace(min(alt), max(alt), n)

X, Y = np.meshgrid(xlins, ylins)
xe[:,:, x_index] = X
xe[:,:, y_index] = Y

ye[:,:,:] = make_predictions(xe.reshape((n**2, nx))).reshape((n, n, ny))
Z = ye[:,:,output_variable].flatten()

Z = Z.reshape(n, n)


#### Layout #####

output_file("layout.html")

## Sample data points ##
x = list(range(11))
y0 = x
y1 = [10 - i for i in x]
y2 = [abs(i - 5) for i in x]


N = 500
x = np.linspace(0, 100, N)
y = np.linspace(0, 100, N)

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = 0 
p.y_range.range_padding = 0
p.plot_width = 500
p.plot_height = 500
p.xaxis.axis_label = "Mach"
p.yaxis.axis_label = "Altitude"
p.min_border_left = 100
# must give a vector of image data for image parameter
p.image(image=[Z], x=0, y=0, dw=max(mach), dh=max(alt), palette="Viridis11")


# create a new plot
s1 = figure(plot_width=200, plot_height=500, y_range=(0,max(alt)), title="Altitude vs Thrust")
s1.xaxis.axis_label = "Thrust"
s1.yaxis.axis_label = "Altitude"
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# create and another
s3 = figure(plot_width=500, plot_height=200, x_range=(0,max(mach)), title='Mach vs Thrust')
s3.xaxis.axis_label = "Mach"
s3.yaxis.axis_label = "Thrust"
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# make a grid
grid = gridplot([[s1, p], [None, s3]])

# show the results
show(grid)






from bokeh.layouts import row, column
from bokeh.models import CustomJS, Slider
from bokeh.plotting import figure, output_file, show, ColumnDataSource

x = np.linspace(0, 10, 500)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

amp_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Amplitude")
freq_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Frequency")
phase_slider = Slider(start=0, end=6.4, value=0, step=.1, title="Phase")
offset_slider = Slider(start=-5, end=5, value=0, step=.1, title="Offset")

callback = CustomJS(args=dict(source=source, amp=amp_slider, freq=freq_slider, phase=phase_slider, offset=offset_slider),
                    code="""
    const data = source.data;
    const A = amp.value;
    const k = freq.value;
    const phi = phase.value;
    const B = offset.value;
    const x = data['x']
    const y = data['y']
    for (var i = 0; i < x.length; i++) {
        y[i] = B + A*Math.sin(k*x[i]+phi);
    }
    source.change.emit();
""")

amp_slider.js_on_change('value', callback)
freq_slider.js_on_change('value', callback)
phase_slider.js_on_change('value', callback)
offset_slider.js_on_change('value', callback)

layout = row(
    plot,
    column(amp_slider, freq_slider, phase_slider, offset_slider),
)