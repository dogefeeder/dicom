from pydicom import dcmread
# from pydicom.filewriter import write_file_meta_info

from pynetdicom import AE, debug_logger, evt, AllStoragePresentationContexts, ALL_TRANSFER_SYNTAXES
from pynetdicom.sop_class import CTImageStorage

import numpy as np
import os, json
from time import time,sleep

# trt
import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt

# ct processing
import nibabel as nib
from skimage.measure import label, regionprops
from PIL import Image
from statistics import mode

todo = './todo'
deskull_engine_path = './CT_Deskull_Single.engine'
denoising_engine_path = './CTDenoise_kelvin.engine'
datatype = 'float16'
img_shape = 512
batch_size = 1
TRT_LOGGER = trt.Logger()
print(trt.__version__)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    #import pycuda.driver as cuda

    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        #import pycuda.driver as cuda
        import pycuda.autoinit
 
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size

        #self.cuda = cuda
        
    def reallocate_context(self):
        #import pycuda.autoinit
        
        #cuda.init()
        #device = cuda.Device(0)
        #ctx = device.make_context()
        #self.context = self.engine.create_execution_context()
        #ctx = cuda.Context.attach()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        #ctx.detach()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        #import pycuda.autoinit
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray,batch_size=1):
#         x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]
   
    #def __del__(self):
    #    pass 
        #del self.inputs
        #del self.outputs
        #del self.stream 
        #self.cuda.Context.pop()

    # deskull
def deskull(orig_):
    print('\n*** deskulling ***')
    # import and preprocessing image
    orig = np.expand_dims(orig_,-1)
    orig[orig>100]=100.
    orig[orig<-100]=-100.
    print(f'Deskull orig shape: {orig.shape}')
    # import engine
    t0 = time()
    #with TrtModel(deskull_engine_path) as model:
    model = TrtModel(deskull_engine_path) 
    t1 = time()
    print(f'Time load engine: {t1-t0} s')
    # engine inference
    results = []
    for i in range(orig.shape[0]):
        model.reallocate_context()
        results.append(0.5<model(orig[i],batch_size)[0][0].reshape([img_shape,img_shape]))
    #cuda.Context.pop()
    Mask_pred = np.array(results,dtype='int8')
    print('Deskulled shape:',Mask_pred.shape)
    # Keep only the largest blob
    label_img = label(Mask_pred)
    # sizes = ndimage.sum(binary_img, range(label_n + 1))
    regions = regionprops(label_img)
    max_blob = np.argmax([props.area for props in regions])
    mask = np.zeros_like(Mask_pred)
    for x,y,z in regions[max_blob].coords:
        mask[x,y,z] = 1
    t2 = time()
    print(f'Time inference: {t2-t1} s')
    orig0 = orig[:,:,:,0].copy()
    orig0[mask==0]=0
    return orig0,np.array(mask,dtype=bool) # deskulled orig, skull==0
    # save image in .nii.gz format
    # skullmask = np.transpose(mask,[1,2,0])
    # x = nib.Nifti1Image(skullmask,PARAMS['affine'], PARAMS['header'])
    # save mask file
    # nib.save(x,datafile.replace('.nii.gz','_mask.nii.gz'))
    # save orig deskulled
    # xtest = np.transpose(x_test_fp16[:,:,:,0],[1,2,0])
    # xtest[skullmask==0]=0
    # x = nib.Nifti1Image(xtest,PARAMS['affine'], PARAMS['header'])
    # x.set_data_dtype(np.double)
    # nib.save(x,datafile.replace('.nii.gz','_deskull.nii.gz'))
    

def denoising(orig_deskulled):
    #global PARAMS
    print('\n*** denoising ***')
    # import and preprocessing image
    x_test = np.expand_dims(orig_deskulled,-1)
    ind = np.logical_and(x_test<=88,x_test>0)
    x_test[x_test>=100]=0
    x_test[x_test<=0]=0
    x_test[x_test>=88]=88
    ctmode=mode(x_test[ind])
    # # Mode correction to 37 HU
    x_test_fp16=x_test.astype(datatype).copy()
    x_test_fp16[ind]=x_test_fp16[ind] - ctmode + 37
    # # Image normalization
    x_test_fp16 -= 33.
    x_test_fp16 /= 11.
    # converting data type
    print(f'image loaded ! Input shape: {x_test_fp16.shape}')
    # import engine
    t0 = time()
    #with TrtModel(denoising_engine_path) as model:
    model = TrtModel(deskull_engine_path) 
    t1 = time()
    print(f'Time load engine: {t1-t0} s')
    # engine inference
    results = []
    for i in range(x_test_fp16.shape[0]):
        model.reallocate_context()
        results.append(model(x_test_fp16[i],batch_size)[0][0].reshape([img_shape,img_shape]))
    results = np.array(results)
    results = results.astype(datatype)
    # results_preundo = np.transpose(results,[1,2,0])
    # save pre-undo CT mode correction 
    # x = nib.Nifti1Image(results_, PARAMS['affine'], PARAMS['header'])
    # x.set_data_dtype(np.double)
    # pre-undo
    # nib.save(x,datafile.replace('_deskull.nii.gz','_denoised_preundo.nii.gz'))
    print('ctmode: %f'%ctmode)

    # Undo the CT_mode correction
    print('ind shape:',ind.shape)
    results = np.expand_dims(results,-1) 
    print('denoising result shape:',results.shape)
    results[ind] = results[ind] -37 + ctmode
    results[~ind] = x_test[~ind].astype('float16')
    print('background mean: ',np.mean(results[~ind]))
    # results = results[:,:,:,0]
    # results_ = np.transpose(results,[1,2,0])
    t2 = time()
    print(f'Time inference: {t2-t1} s\n')
    # save image in .nii.gz format
    # x = nib.Nifti1Image(results_,PARAMS['affine'], PARAMS['header'])
    # x.set_data_dtype(np.double)
    # nib.save(x,datafile.replace('_deskull.nii.gz','_denoised.nii.gz'))
    return results[:,:,:,0] # denoised without bone

def combine(orig,mask,denoised_):
    denoised = denoised_.copy()
    denoised[mask] = orig[mask].copy()
    return denoised # denoised with bone
    # x = nib.Nifti1Image(denoised.astype('float16'), affine, header)
    # x.set_data_dtype(np.double)
    # nib.save(x,file_info['filepath'].replace('.nii.gz','_output.nii.gz'))  

def sort_key(a):
    return int(a.split('.')[-1])

if __name__ == '__main__':
    n_fps = 0
    while True:
        fps = [os.path.join(todo,f) for f in os.listdir(todo) if f[0]!='.']
        #print(fps)
        if len(fps)==0:
            sleep(2)
            continue
        if n_fps<len(fps):
            n_fps = len(fps)
            sleep(2)
            continue
        if n_fps == len(fps):
            ae = AE()
            ae.add_requested_context(CTImageStorage)
            assoc = ae.associate('smab-lambda-dual', 11112)
            if not assoc.is_established:
                print('Association rejected, aborted or never connected')
            
            else:
                fps = sorted(fps,key=sort_key)
                dss = []
                orig = []
                for f in fps:
                    dss.append(dcmread(f))
                    orig.append(dss[-1].pixel_array)
                orig = np.array(orig)
                
                output,mask = deskull(orig)
                #print('1.',orig_deskulled.shape,mask.shape)
                output = denoising(output)
                #print('2.',output.shape, mask.dtype)
                #orig = np.array(orig,dtype=np.float16)
                print(orig.shape,orig.dtype)
                print(mask.shape,mask.dtype)
                print(output.shape,output.dtype)
                output[mask] = orig[mask]
                #output = combine(orig,mask,orig_denoised)
                #print('3.',output.shape)
                

                for i in range(len(dss)):
                    dss[i].PixelData = output[i].tobytes()
                
                for ds in dss:
                    status = assoc.send_c_store(ds)
                    # Check the status of the storage request
                    if status:
                        # If the storage request succeeded this will be 0x0000
                        print('C-STORE request status: 0x{0:04x}'.format(status.Status))
                    else:
                        print('Connection timed out, was aborted or received invalid response')
            assoc.release()
            dss = []
            orig = []
            for f in fps:
                os.remove(f)
        sleep(3)
            
            




