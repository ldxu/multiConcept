"""
    multiConcept的注意力交换代码
    photoswap也可以公用这部分代码
"""

from typing import List, Union, Tuple, Dict, Optional
import abc
import torch.nn.functional as nnf
import torch

from utils import get_refinement_mapper
import utils 

class LocalBlend:
    
    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        # [2, 40, 1, 16, 16, 77]*[2, 1, 1 ,1 ,1 ,77]
        maps = (maps * alpha).sum(-1).mean(1)
        # 提取目标subject token的交叉注意力图
        # maps->[2, 1, 16 ,16 ] 
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        # utils.show_cross_attention_single(mask)
        # 上采样将注意力图放大到与噪声图像相同大小
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        # 取数据中的最大值进行归一化处理
        # 取大于阈值部分的特征图
        mask = mask.gt(self.th[1-int(use_pool)])
        # 找到源图中Class Token对应的高分数区(即目标主题主要区域)，将源图区域覆盖Target Token的区域
        # mask = mask[:1] + mask
        mask[1] = mask[0]
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            # maps[0]->torch.Size([16, 256, 77])
            # 提取分辨率为16*16大小的交叉注意力图
            # len(maps)->5
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            # maps[0]->Troch.Size([2,8,1,16,16,77])
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, 77) for item in maps]
            # maps->torc.Size([2, 40, 1, 16, 16, 77])
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t, maps, self.alpha_layers, False)
            if self.substruct_layers is not None:
                maps_sub = self.get_mask(x_t, maps, self.substruct_layers, True)
                mask = mask + maps_sub
            mask = mask.float()
            # 将Class Token交叉注意力图中的区域扩散到噪声图像中 保留原有背景噪声结果
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, device, NUM_DDIM_STEPS,
                 substruct_words=None, start_blend=0.2, th=(.3, .3)):
        # alpha_layers -> [2, 1, 1, 1, 1, 77]
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 77)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = utils.get_word_inds(prompt, word, tokenizer)
                # 只有与words中对应的token的数据为1
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 77)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        # 只在部分阶段进行local blend
        self.counter = 0 
        self.th=th
        

        
class AttentionControlEdit(abc.ABC):
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
        
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
            
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer, device):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
        self.batch_size = len(prompts)
        self.cross_replace_alpha = utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

        
class AttentionSwap(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_map_replace_steps: float, self_map_replace_steps: float, self_output_replace_steps: float,
                 source_subject_word=None, target_subject_word=None, tokenizer=None, device=None, LOW_RESOURCE=False, substruct_words = None):
        self_map_replace_steps = self_map_replace_steps + self_output_replace_steps
        blend_word = (((source_subject_word,), (target_subject_word,)))
        local_blend = LocalBlend(prompts, blend_word, tokenizer, device, num_steps, substruct_words=substruct_words)
        
        super(AttentionSwap, self).__init__(prompts, num_steps, cross_map_replace_steps, self_map_replace_steps, local_blend, tokenizer, device)
        self.cross_map_replace_steps = cross_map_replace_steps
        self.self_map_replace_steps = self_map_replace_steps
        self.self_output_replace_steps = self_output_replace_steps
        
        self.mapper, alphas = get_refinement_mapper(prompts, tokenizer)
        # alphas -> [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, ...] Target Prompt与源Prompt对应的序列，0代表多出来的token
        # mapper -> [0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9, 10...] 其中-1代表多出来的token
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.LOW_RESOURCE = LOW_RESOURCE
