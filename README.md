**Steps:**

0. Download data from `https://drive.google.com/file/d/1g_ixtP0aehUQ-Z7vz9QmB5MyfiiUDVPF/view?usp=sharing` (I'll upload zip version tomm)
1. Add data called `train_X_y_ver_all_xyz_energy.pt` into `/data` folder
2. Install packages `pip install -r requirements.txt` (Should work but haven't tested. Let me know if it doesn't)
3. run `. multiple_exp.sh` (edit bash file to change configuration such as adding --debug to run **sample** of data)
4. See if you can replicate plots inside `example_plots` folder

**Notes:**
- Full training should take ~20 min (based on 4 x A5000)
- This code is agnostic of device (gpu, cpu, multi-gpu)

**Example Plots:**

Fully Trained Plot
![Example Image](example_plots/pointNET_hist.png)

Debug Mode Plot
![Example Image](example_plots/debug_pointNET_hist.png)
