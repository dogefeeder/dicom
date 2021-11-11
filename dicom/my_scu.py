import os

from pydicom import dcmread
from pydicom.filewriter import write_file_meta_info

from pynetdicom import AE, debug_logger, evt, AllStoragePresentationContexts, ALL_TRANSFER_SYNTAXES
from pynetdicom.sop_class import CTImageStorage
fp = './dcms/case1'
# import SimpleITK as sitk

debug_logger()

# Initialise the Application Entity
ae = AE()

# Add a requested presentation context
ae.add_requested_context(CTImageStorage)
# for scp
storage_sop_classes = [
    cx.abstract_syntax for cx in AllStoragePresentationContexts
]
for uid in storage_sop_classes:
    ae.add_supported_context(uid, ALL_TRANSFER_SYNTAXES)


# Read in our DICOM CT dataset
#ds = dcmread('./dcms/case1')
# ds = sitk.ReadImage('./l1_Orig.nii.gz')

# Associate with peer AE at IP 127.0.0.1 and port 11112
#assoc = ae.associate('jetson-agx', 11112)
assoc = ae.associate('jetson-agx', 11112)
if assoc.is_established:

    # Use the C-STORE service to send the dataset
    # returns the response status as a pydicom Dataset
    fps = [os.path.join(fp,f) for f in os.listdir(fp) if f[0]!='.']
    for f in fps:
        ds = dcmread(f)
        status = assoc.send_c_store(ds)
    
    # Check the status of the storage request
    if status:
        # If the storage request succeeded this will be 0x0000
        print('C-STORE request status: 0x{0:04x}'.format(status.Status))
    else:
        print('Connection timed out, was aborted or received invalid response')

    # Release the association
    #assoc.release()
else:
    print('Association rejected, aborted or never connected')


# scp
def handle_store(event, storage_dir):
    """Handle EVT_C_STORE events."""
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except:
        # Unable to create output dir, return failure status
        return 0xC001

    # We rely on the UID from the C-STORE request instead of decoding
    fname = os.path.join(storage_dir, event.request.AffectedSOPInstanceUID)
    with open(fname, 'wb') as f:
        # Write the preamble, prefix and file meta information elements
        f.write(b'\x00' * 128)
        f.write(b'DICM')
        write_file_meta_info(f, event.file_meta)
        # Write the raw encoded dataset
        f.write(event.request.DataSet.getvalue())
    return 0x0000

handlers = [(evt.EVT_C_STORE, handle_store, ['received'])]
ae.start_server(('', 11112), block=True, evt_handlers=handlers)
