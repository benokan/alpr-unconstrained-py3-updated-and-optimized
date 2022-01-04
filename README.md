# alpr-unconstrained-py3-updated-and-optimized

Updated a little bit the already existing alpr-unconstrained repository. https://github.com/sergiomsilva/alpr-unconstrained

Also since keras model for wpod is working way slower than the darknet models it was creating a bottleneck for the inference, to avoid this problem I've introduced a trade-off by changing the resolution of the input for wpod which doesnt affect the performance of the model if yoou don't over-tweak it (as setting the resolution to 10 would destroy the inference)

Solved compatability issues between old darknet and the new one for this repository. I've compiled darknet using cmake-gui in Windows from AlexeyAB's repository https://github.com/AlexeyAB/darknet

A tip would be first compile your darknet then change it with my darknet folder or change directly darknet.py folder itself.

And for both license plate (ocr) and vehicle detection added confidence scores for each detection.(avg for ocr)

You can download the weights and cfg files from the original repository.

# A screenshot from demo video

![Alt text](1_output.png?raw=true "Screenshot")
*The reason it didn't capture the car at bottom left is the full car is not in the frame yet and in the following frames yolo actually finds it and continues with wpod and ocr expectedly*
