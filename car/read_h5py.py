import h5py
filename = "C:\\Users\\majic\\OneDrive\\바탕 화면\\capstone_final-master\\car_license_plate_recognition\\car\\weight\\test3.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
