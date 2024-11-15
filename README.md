# LAT Alignment
Tools for Large Aperture Telescope (LAT) mirror alignment.
While these tools are currently specific to the Simons Observatory LAT,
much of the library could be generalized to any telescope alignment.

For details on usage please check the guide (WIP! Doesn't exist yet!).

## Installation
Technically after cloning this repository you can just run `python lat_alignment/alignment.py PATH/TO/CONFIG`,
but it is recommended that you install this as a package instead.

To do this just run: `pip install -e .` from the root of this repository.

This has two main benefits over running the script directly:
1. It will handle dependencies for you.
2. This sets up an entrypoint called `lat_alignment` so that you can call the code from anywhere.
This is nice because now you can call the code from the measurement directory where you are most likely editing files,
saving you the hassle of having to `cd` or wrangle long file paths.

## Bugs and Feature Requests
For low priority bugs and feature requests submit an issue on the [git repo](https://github.com/simonsobs/LAT_Alignment).

For higher priority issues (or questions that require an expedient answer) [email](mailto:haridas@sas.upenn.edu), Slack, or call me.

## Contributing
If you wish to contribute to this repository (either code or adding measurement files) contact me via [email](mailto:haridas@sas.upenn.edu) or Slack.

If you are contributing code please do so by creating a branch and submitting a pull request. Try to keep things as close to PEP8 as possible.
