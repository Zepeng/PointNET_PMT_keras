# Custom inputs for RFSoC4x2 `deepsocflow`
Note: `deepsocflow` is not currently setup to run custom inputs, and this process will require backend changes to the library

1. Make modifications to `deepsocflow`. In `deepsocflow/py/xmodel.py`, make the following changes to the `export_inference` function
```
def export_inference(model, hw, custom_input):
    ...
    input_shape = (1, *model.inputs[0].shape[1:]) # this is same
    x_keras = custom_input # this is different
    x_qtensor = user_model.input_quant_layer(x_keras)
    ...
```
Be sure to run `pip install .` on the `deepsocflow` library again to install these changes. Note that the shape of the custom input should be of the form of `input_shape`, though the 1 can be changed to any batch size that is smaller than your allowed max batch size defined by the `hw` parameter

2. Now export your model as normal using the `deepsocflow` library, though I'd recommend using a smaller batch size for the initial export, since the simulator will then have to process a large `wbx.bin` file, which can take a long time or use 100's of GB of storage.

3. After exporting your model and synthesizing the design, you may want to export using a larger batch size **without simulation**. The generated `wbx.bin` file will be in the `vectors/` folder.

4. Using Vitis, load the model and the custom input as prescribed by [this video (at a timestamp)](https://youtu.be/cWdDxEbrMuA?feature=shared&t=266).

5. deepsocflow will print the model's output word by word, which means a four-dimensional output will take the form:

```
y[0]: ...
y[1]: ...
y[2]: ...
y[3]: ...
```
So, if you have a batch size of two, you'd expect eight lines of output in this case.

6. If your model does not have any output integer bits, you will need to divide each output from the model by `2^(SYS_BITS.X - 1 - integer_bits)`. In the case of using 8 bits for x (defined by the SYS_BITS variable) and no integer bits, this would be `2^7`. This step is necessary since `deepsocflow` computes using integers (for performance), so model outputs that are purely fractional will appear as integers on the fpga output.