import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import click
import transformers
from transformers import AutoProcessor, AutoModel
from dataset_vl_bench.data import Dataset_v1
from dataset_vl_bench.utils import process_path

from src.otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration
#from src.otter_ai.models.otter.modeling_otter import OtterModel


MODELS = ("luodian/OTTER-9B-DenseCaption",)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len > converted_len:
        end_idx = np.random.randint(converted_len, seg_len)
    else:
        end_idx = min(converted_len, seg_len)-1
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


@click.command()
@click.option(
    "-i", "--input-file", type=click.Path(exists=True, file_okay=True), required=True
)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(choices=MODELS),
    default=MODELS[0],
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
)
@click.option(
    '--num-workers',
    type=int,
    default=5,
)
@click.option(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
@click.option(
    "--quva-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "--something-something-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '--youtube-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '--star-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--proficiency',
    is_flag=True,
)
@click.option("--mask-video", type=bool, required=True, default=False)
def main(
    input_file,
    model_name,
    batch_size,
    num_workers,
    device,
    quva_dir,
    something_something_dir,
    youtube_dir,
    star_dir,
    output_file,
    proficiency,
    mask_video,
):
    print(f"- running xclip on {input_file}")
    print(f"- output to {output_file}")
    # check video datasets' dirs
    assert quva_dir is not None \
        or something_something_dir is not None \
        or youtube_dir is not None
    if quva_dir is not None:
        quva_dir = process_path(quva_dir)
    if something_something_dir is not None:
        something_something_dir = process_path(something_something_dir)
    if youtube_dir is not None:
        youtube_dir = process_path(youtube_dir)
    if star_dir is not None:
        star_dir = process_path(star_dir)
    np.random.seed(0)

    # initialize model & processor
    load_bit = "fp32"
    if load_bit == "fp16":
        precision = {"torch_dtype": torch.float16}
    elif load_bit == "bf16":
        precision = {"torch_dtype": torch.bfloat16}
    elif load_bit == "fp32":
        precision = {"torch_dtype": torch.float32}

    # This model version is trained on MIMIC-IT DC dataset.
    model = OtterForConditionalGeneration.from_pretrained(
        "luodian/OTTER-9B-DenseCaption", 
        device_map="cuda", 
        **precision
    )
    tensor_dtype = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32
    }[load_bit]
    #model = OtterModel.from_pretrained("luodian/otter-9b-hf", device_map="cuda", **precision)

    # processors
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    # loss
    crit = nn.CrossEntropyLoss(
        reduction='none',
        ignore_index=tokenizer.pad_token_id,
    )

    # read data
    data_v1 = Dataset_v1(
        input_file,
        quva_dir=quva_dir,
        something_something_dir=something_something_dir,
        youtube_dir=youtube_dir,
        star_dir=star_dir,
        proficiency=proficiency,
    )

    results = dict()
    for item in tqdm(data_v1):

        video_len = item['video'].shape[0]
        clip_len = 8  # FIXME: hardcoded
        downsampled = item['video']
        if video_len > clip_len:
            indices = sample_frame_indices(
                clip_len=clip_len,
                frame_sample_rate=1,
                seg_len=item['video'].shape[0],
            )
            downsampled = item['video'][indices]
        num_texts = len(item['raw_texts'])
        num_frames = len(downsampled)

        # pre-process video
        downsampled = image_processor.preprocess(downsampled, return_tensors="pt")["pixel_values"]
        downsampled = downsampled.half()
        downsampled = downsampled.unsqueeze(1).repeat_interleave(num_texts, dim=1)
        C, H, W = downsampled.shape[2:]
        downsampled = downsampled.reshape(-1, C, H, W)
        downsampled = downsampled.unsqueeze(0).unsqueeze(0)

        # pre-process text
        print(f"raw_texts: {item['raw_texts']}")
        lang_x = model.text_tokenizer(item['raw_texts'], return_tensors="pt", padding=True)
        lang_x["input_ids"] = lang_x["input_ids"].unsqueeze(0).repeat_interleave(num_frames, dim=0)
        lang_x['input_ids'] = lang_x['input_ids'].reshape(-1, lang_x['input_ids'].shape[-1])
        
        # pre-process mask
        lang_x['attention_mask'] = lang_x['attention_mask'].unsqueeze(0).repeat_interleave(num_frames, dim=0)
        lang_x['attention_mask'] = lang_x['attention_mask'].reshape(-1, lang_x['attention_mask'].shape[-1])

        # get logits
        with torch.no_grad():
            output = model(
                vision_x=downsampled.to(model.device, dtype=tensor_dtype), 
                lang_x=lang_x["input_ids"].to(model.device), 
                attention_mask=lang_x["attention_mask"].to(model.device)
            )
        
        # get scores as avg. ppl
        logits = output.logits[:, :-1, :]
        labels = lang_x['input_ids'][:, 1:].contiguous()
        lengths = lang_x['attention_mask'].sum(dim=-1)
        scores = crit(logits.reshape(-1, logits.shape[-1]), labels.view(-1).to(logits.device))
        scores = scores.reshape_as(labels)
        scores = scores.sum(dim=1) / lengths.to(scores.device)
        scores = scores.reshape(num_frames, num_texts)
        scores = scores.mean(dim=0).exp().tolist()
        item_id = item['item_id']
        results[item_id] = {'scores': scores}

    # save results 
    with open(process_path(output_file), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()