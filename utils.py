import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm


class ScoreParams:
    """
    两者相等则为1,不然则为-1
    """

    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    """
        example: "soup" -> "pea soup"
        return mappers-> tensor([[ 0, -1,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
         72, 73, 74, 75, 76]], device='cuda:0')

               alphas->tensor([[1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1.]], device='cuda:0')

         得到映射序列
    """
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


def get_word_inds(text: str, word_place: int, tokenizer):
    """
        得到prompt中对应于需要编辑目标在经过Text Model编码后的index
        例:['a', 'painting', 'of', 'a', 'squirrel', 'eating', 'a', 'lasag', 'ne']
        存在的问题:有些单词可能在编码的时候可能会出现两个index,比如例子中的lasagne,在进行编码时
        会被拆成lasag 和 ne, 从而出现有两个index的情况
        该函数可以找出在编码器中subject对应的index,因为我们最终是需要进行text encoder
        得到相应的token嵌入,所以需要进行这一步操作
        原因:我们正常使用的句子与编码器中的会存在差异,编码器会对一些单词进行拆分编码
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        # 找到替换的token在Prompt中的位置
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        # 首先编码Prompt然后进行解码，去头去尾，过滤掉Start 以及 End
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)




def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds

    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    """
    将多个图像组合到一个图像中显示
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    # offset = 10
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)

def show_cross_attention_single(img , counter:int=0, compare:bool=False, threshold=0.1):
    images = []
    opt_img = []
    # [2, 1, 64, 64]
    # if compare:
    #     opt_img = img.clone()

    #     opt_img[0] = region_growing(opt_img[0], threshold)
    #     opt_img[1] = region_growing(opt_img[1], threshold)
    for i in range(len(img)):
        # 使用 clone() 创建原始 Tensor 的副本
        if compare:
            image = img[i].permute(1, 2, 0).cpu()
            image = 255 * image / image.max()

            image_opt = opt_img[i].permute(1,2,0).cpu()
            # image_opt = 255 * image_opt / image_opt.max()
            image_opt = image_opt * image
            image_opt = 255 * image_opt / image_opt.max()
            # 将单通道灰度图像转化为三通道图像->[16, 16, 3]
            image = image.expand(64,64, 3)
            image = image.numpy().astype(np.uint8)
            
            image_opt = image_opt.expand(64,64, 3)
            image_opt = image_opt.numpy().astype(np.uint8)

            # 使用 Image.fromarray 创建图像对象，并调整大小
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image_opt = np.array(Image.fromarray(image_opt).resize((256, 256)))
            if  i==0:
                text_1 = "base_origin" + str(counter)
                text_2 = "base_opt" + str(counter)
                image = text_under_image(image,text_1)
                images.append(image)
                image_opt = text_under_image(image_opt,text_2)
                images.append(image_opt)
            else:
                text_1 = "target_origin" + str(counter)
                text_2 = "target_opt" + str(counter)
                image = text_under_image(image,text_1)
                images.append(image)
                image_opt = text_under_image(image_opt,text_2)
                images.append(image_opt)
        else:
            image = img[i].permute(1, 2, 0).cpu()
            image = 255 * image / image.max()
            # 将单通道灰度图像转化为三通道图像->[16, 16, 3]
            image = image.expand(64,64, 3)
            image = image.numpy().astype(np.uint8)
            # 使用 Image.fromarray 创建图像对象，并调整大小
            image = np.array(Image.fromarray(image).resize((256, 256)))
            if  i==0:
                text = "base_origin" + str(counter)
                image = text_under_image(image,text)
                images.append(image)
            else:
                text = "target_origin" + str(counter)
                image = text_under_image(image,text)
                images.append(image)
        # 给展示的图像加底部文字
    
    # ptp_utils 模块未提供，你可能需要自己替换成实际的图像显示方法
    # 以下是一个示例，使用 matplotlib.pyplot 来显示图像
    view_images(images)
    return opt_img

