# Configuration File

## Fields

???+ info "`mode`"

    The alignment mode to use.

    Possible values are:

    * `panel`: For aligning panels within one mirror.
    * `optical`: For aligning optical elements relative to each other (as solid bodies).

???+ info "`mirror`"

    The mirror that we want to align panels to.
    Only used if `mode` is `panel`.

    Possible values are:

    * `primary`: To align the primary mirror.
    * `secondary`: To align the secondary mirror.

???+ info "`align_to`"

    Which optical element to keep fixed and align the others to.
    Only used if `mode` is `optical`.

    Possible values are:

    * `primary`: To align to the primary mirror.
    * `secondary`: To align to the secondary mirror.
    * `receiver`: To align to the receiver.
    * `bearing`: To align to the bearing.

???+ info "`measurement`"

    The path to the photogrammetry data we are using to do the alignment.
    If this is a relative path it is taken relative to the directory that the
    configuration file is in.

???+ info "`data_dir`"

    The path to the data files that define the panel corners and the adjuster positions.
    If this is a relative path it is taken relative to the directory that the
    configuration file is in.

    You genrally don't need to provide this since the package will use its own bundled
    data files by default.

???+ info "`load`"

    Additional keyword arguments to pass to
    [`io.load_photo`](https://simonsobs.github.io/LAT_Alignment/latest/reference/io/#lat_alignment.io.load_photo).


???+ info "`load`"

    Additional keyword arguments to pass to
    [`photogrammetry.align_photo`](https://simonsobs.github.io/LAT_Alignment/latest/reference/photogrammetry/#lat_alignment.photogrammetry.align_photo).


???+ info "`compensate`"

    Amount to compensate mirror measurements by in mm.
    This is for backwards compatiblilty with laser tracker data and is $0$ by default.


???+ info "`common_mode`"

    Additional keyword arguments to pass to
    [`mirror.remove_cm`](https://simonsobs.github.io/LAT_Alignment/latest/reference/mirror/#lat_alignment.mirror.remove_cm)

???+ info "`adjuster_radius`"

    How close to an adjuster a data point needs to be in order for us to use its residual as
    a secondary correction when computing adjustments.
    Only used if `mode` is `panel`.

    This is $100$ mm by default.

???+ info "`vmax`"

    The maximum value to use in the colorbar when plotting mirror surface.
    The colorbar is symmetric so `vmin = -1*vmax`.

???+ info "`adjust`"

    Additional keyword arguments to pass to
    [`adjustments.calc_adjustments`](https://simonsobs.github.io/LAT_Alignment/latest/reference/adjustments/#lat_alignment.adjustments.calc_adjustments)

???+ info "`title`"

    The title of the measurement.
    This is used both in plots and in output filenames.


## Example Configuration Files
These are typical configuration files,
you usually will not need to touch fields other than the ones shown here.

### Panel Alignment

```yaml
mode: "panel"
mirror: "secondary"
measurement: "data_20240911_1430.csv"
title: "M2 20240911 1430"
vmax: 50
```

### Optical Element Alignment

WIP! Check back later!
