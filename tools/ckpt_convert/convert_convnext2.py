import torch

def convert(path):
    ckpt = torch.load(path, map_location='cpu')
    new_ckpt = ckpt['state_dict']
    # if list(new_ckpt.keys())[0].startswith('backbone.'):
    new_state_dict = {k[9:]: v for k, v in new_ckpt.items() if k.startswith('backbone.')}

    # fcmae
    # for k, v in new_ckpt.items():
    #     if 'grn' in k:
    #         new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    # import pdb
    # pdb.set_trace()
    torch.save(new_state_dict, path)

if __name__ == '__main__':
    path = '/home/oneco/xyh/rotate_detection/checkpoints/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'
    convert(path)