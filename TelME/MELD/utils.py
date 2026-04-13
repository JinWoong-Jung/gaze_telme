import torch
from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoImageProcessor
import librosa
import cv2
import numpy as np

audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.mask_token_id]

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [ 1 for _ in range(len(ids))]
        add_attention = [ 0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids+ids)
        attention_masks.append(add_attention+attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def padding_video(batch):
    max_len = 0
    for ids in batch:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in batch:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids.tolist())
    
    return torch.tensor(pad_ids)

def get_audio(processor, path):
    audio, rate = librosa.load(path)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    return inputs["input_values"][0]
    
def get_video(feature_extractor, path):

    video = cv2.VideoCapture(path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    step = length // 8
    count = 0
    if length >= 8:

        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break

            count += 1
            if count % step == 0:
                frames.append(image)
        video.release()

    else:
        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break

            frames.append(image)

        video.release()
        lack = 8 - len(frames)
        extend_frames = [ frames[-1].copy() for _ in range(lack)]
        frames.extend(extend_frames)

    inputs = feature_extractor(frames[:8], return_tensors="pt")

    return inputs["pixel_values"][0]

def make_batchs(sessions):
    """Collate sessions into model-ready tensors.

    Supports two session formats:
      Original (use_gaze=False): each turn = [speaker, utt, video_path, emotion]
      Gaze-augmented (use_gaze=True): each turn = [speaker, utt, video_path, emotion, gaze_vec]

    Returns:
      (batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels)
      — gaze-mode appends batch_gaze as the 6th element.
    """
    import numpy as np

    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    batch_gaze = []
    max_length = 400000

    # Detect gaze mode from first turn of first session
    use_gaze = (len(sessions[0][0]) == 5) if sessions else False

    for session in sessions:
        inputString = ""
        now_speaker = None
        last_gaze = np.zeros(6, dtype=np.float32)

        for turn, line in enumerate(session):
            if use_gaze:
                speaker, utt, video_path, emotion, gaze_vec = line
                last_gaze = np.array(gaze_vec, dtype=np.float32)
            else:
                speaker, utt, video_path, emotion = line

            # text
            inputString += '<s' + str(speaker + 1) + '> '
            inputString += utt + " "
            now_speaker = speaker

        audio, rate = librosa.load(video_path)
        duration = librosa.get_duration(y=audio, sr=rate)
        if duration > 30:
            batch_video.append(torch.zeros([8, 3, 224, 224]))
            batch_audio.append(torch.zeros([1412]))
        else:
            audio_input = get_audio(audio_processor, video_path)
            audio_input = audio_input[-max_length:]
            batch_audio.append(audio_input)

            video_input = get_video(video_processor, video_path)
            batch_video.append(video_input)

        prompt = "Now" + ' <s' + str(now_speaker + 1) + '> ' + "feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        label_ind = label_list.index(emotion)
        batch_labels.append(label_ind)

        if use_gaze:
            batch_gaze.append(torch.from_numpy(last_gaze))

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio  = padding_video(batch_audio)
    batch_video  = torch.stack(batch_video)
    batch_labels = torch.tensor(batch_labels)

    if use_gaze:
        batch_gaze_t = torch.stack(batch_gaze)   # (B, 6)
        return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels, batch_gaze_t

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels
