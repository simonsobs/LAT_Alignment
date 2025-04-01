# IxB Instructions

Manually adjusting every mirror panel can be extremely time consuming and error prone.
Here we discuss the setup and use of a [Atlas Copco IxB](https://www.atlascopco.com/en-us/itba/industry-solutions/tighteningsolutionsandservices/tensor-ixb) tool to do this.

This is done using the Atlas Copco Open Protocol, it may be useful to familiarize yourself with the [specification](./pdfs/OpenProtocol_Specification_R_2.20.1.pdf) and [appendix](./pdfs/OpenProtocol_Appendix_SW_3.10.pdf) but note that these are **not** up to date and working with the protocol may require a mix of trail and error and packet sniffing.

## Initial Setup

1. Connect to the tool, this can be done via wifi or USB. If you connect via USB the IP will be `168.254.1.1`, if you connect to the tool's local wifi network the IP should be `192.168.1.1`. To access the web interface simply enter the IP into your browser.
2. Install the backup of the IxB configuration with all of the needed mapping preloaded (contact Saianeesh for this). See [this page](https://picontent.atlascopco.com/cont/external/short/html/IXB_Software/en-US/1419699508326801702027.html_) for instructions on how to do this import.
3. Make sure open protocol is active in the virtual station.

## Sending Data to the Tool

!!! tip "Future Feature Alert!"

    Right now there are limitations of the IxB firmware that prevent your from
    having all adjustors programmed at once! Ongoing development will eventually
    enable a "server-client" mode that will work around this limitation.

Once you have calculated your required adjustments for the mirror you can send it to the tool by running:

```
ixb_send PATH_TO_ADJUSTEMNTS MIRROR_HALF
```

where `PATH_TO_ADJUSTEMNTS` is the path to the output from `lat_alignment` in panel mode
and `MIRROR_HALF` is the half of the mirror we are adjusting.
If `MIRROR_HALF` is `1` then we are adjusting the panels in rows 1-4 and if it in `2`
then we are adjusting rows 5-9.

This command will also print out a list of adjusters above the adjustment threshold,
it is useful to take note of this so you can skip areas that don't need adjusting.
The adjustment threshold and the number of microns per turn of the adjuster can be set as
command line arguments, run `ixb_send --help` for syntax. The IP and port of the tool
can also be specified this way.

Once this is done reboot the IxB tool by removing the battery and waiting for the screen to shut off
and then reinstalling the battery. This is needed ensure that the changes to the tightening program propagate.

The final step is to make sure the correct task is selected, to do this:

1. Connect to the web UI for the IxB tool.
2. Go to the "Integrated Controller Tool" menu
3. Click "Virtual Station"
4. Under "Task" select "Choose Test"
5. Clock on the "Sources" tab
6. Select `Mirror_Part1` or `Mirror_Part` depending on which half of the mirror you are adjusting.
7. Click "Save"

## Using the IxB Tool to Adjust Panels

This part is easy (hopefully):

1. Scan the barcode next to the adjuster. This can be done either by pulling the trigger or hitting the button on top of the scanner.
2. If you have already scanned this adjuster a pop up will appear on the screen asking you to confirm.
3. Place the tool on the edge of the adjuster and hold down the trigger until the screen goes green.
4. Try not to hit your head as you move to the next adjuster.

It will save time if you use the list of adjusters above threshold that `ixb_send` generates to only scan adjuster you need to hit.
