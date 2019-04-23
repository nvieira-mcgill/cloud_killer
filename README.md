# cloud_killer
Software to attempt to remove the effects of cloud coverage from albedo maps of Earth using Markov chain Monte-Carlo methods. Developed under the supervision of Professor N. Cowan at McGill University from Jan-April 2019. Credit also goes to McGill students Elisa Jacquet and Juliette Geoffrion for their work on previous versions of this project.

# Modules you'll need 
This software makes use of certain modules you may not already have. These include:

netCDF4.Dataset -- For loading in the EPIC data.

emcee -- "The MCMC Hammer." Developed by D. Foreman-Mackey et al. Used for all of the Markov chain Monte-Carlo in these scripts.

corner -- Software used to produce the corner plots needed to understand MCMC results. Also developed by D. Foreman-Mackey.

cartopy -- Software for preparing all sorts of maps. Used here to plot albedo maps over an actual map of Earth. 

astropy (specifically, astropy.time.Time) -- Widely-used software for astrophysics-related programming. Just download the whole thing. 

# Where to get these modules:
netCDF4: http://unidata.github.io/netcdf4-python/netCDF4/index.html

emcee: http://dfm.io/emcee/current/#user-guide

corner: https://corner.readthedocs.io/en/latest/install.html

cartopy: https://scitools.org.uk/cartopy/docs/latest/installing.html#installing

astropy: http://docs.astropy.org/en/stable/install.html

# How to use this software

First, you'll need the EPIC data itself. This can be provided by Professor Nicolas Cowan. 

Documentation and a step-by-step guide on the use of this software can be seen here: https://github.com/nvieira-mcgill/cloud_killer/wiki#how-to-use-this-software

# Known bugs 
