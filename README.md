## Dicom
The dicom project includes 3 python files:
1. my_scu.py
The script acts as sender (requestor) of dicom files and receiver (acceptor) of backward dicom files from my_scp.py
2. my_scp.py
The script accepts and stores the dcms sent from my_scu.py.
3. denoising.py
The script loads dcms files received, denoises by case, sends back to my_scu.py and deletes the processed files.
