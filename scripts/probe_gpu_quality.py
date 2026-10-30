"""仅检查PyNvVideoCodec输出接口，不用于测速。"""
import torch
import PyNvVideoCodec as nvc
torch.cuda.set_device(2)
p='datasets/hifi_umi/chunk-0000/part-0000/videos/observation.images.head_main/chunk-000/file-000.mp4'
d=nvc.CreateDemuxer(p)
print('DEMUX',[(k,getattr(d,k)()) for k in ['Width','Height','FrameRate','Timebase'] if hasattr(d,k)],flush=True)
print('DEMUX METHODS',dir(d),flush=True)
dec=nvc.CreateDecoder(gpuid=2,codec=d.GetNvCodecId(),cudastream=torch.cuda.current_stream().cuda_stream,usedevicememory=True)
for packet in d:
    frames=dec.Decode(packet)
    if frames:
        f=frames[0];print('FRAME METHODS',dir(f),flush=True)
        print('TENSOR',torch.from_dlpack(f).shape,torch.from_dlpack(f).dtype,torch.from_dlpack(f).device,flush=True)
        for key in ['timestamp','pts','getPTS','GetPTS','getTimestamp','get_frame_size','format']:
            if hasattr(f,key):
                v=getattr(f,key);print(key,v() if callable(v) else v,flush=True)
        print('PACKET',[(k,getattr(packet,k)) for k in dir(packet) if not k.startswith('_')],flush=True)
        break
