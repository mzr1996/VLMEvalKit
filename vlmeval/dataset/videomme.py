import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         get_logger, load, md5, modelscope_flag_set)
from vlmeval.utils import track_progress_rich
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

logger = get_logger(__name__)

FAIL_MSG = 'Failed to obtain answer via API.'


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


def _evaluate_single(line, model):
    """Evaluate a single VideoMME sample, supporting both LLM Judge and regex extraction."""
    from .utils.videomme import extract_characters_regex, extract_option, llm_judge_mcq
    pred = str(line['prediction'])

    if model is not None:
        # Use LLM Judge for evaluation
        options = eval(line['candidates'])
        extracted = llm_judge_mcq(model, line['question'], options, pred)
    else:
        # Fallback to regex extraction
        regex_result = extract_characters_regex(pred)
        if regex_result == '':
            extracted = extract_option(
                model,
                line.to_dict(),
                'Video-MME'
            )
        else:
            extracted = regex_result

    return {'index': line['index'], 'extracted': extracted}


class VideoMME(VideoBaseDataset):

    MD5 = '85bdd91f9b29a99354c23b97ab7c113c'
    SYS = ''

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='Video-MME', use_subtitle=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['Video-MME']

    def prepare_dataset(self, dataset_name='Video-MME', repo_id='lmms-lab/Video-MME'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('video')
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

                subtitle_zip_file = os.path.join(base_dir, 'subtitle.zip')
                subtitle_target_dir = os.path.join(base_dir, 'subtitle')

                if not os.path.exists(subtitle_target_dir):
                    os.makedirs(subtitle_target_dir, exist_ok=True)
                    with zipfile.ZipFile(subtitle_zip_file, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # Check if the member is a file (not a directory)
                            if not member.endswith('/'):
                                # Extract the file to the specified directory
                                source = zip_ref.open(member)
                                target = open(os.path.join(subtitle_target_dir, os.path.basename(member)), 'wb')
                                with source, target:
                                    target.write(source.read())
                    print('The subtitle file has been restored and stored from the zip file.')
                else:
                    print('The subtitle file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'videomme/test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video'] = data_file['videoID']
                data_file['video_path'] = data_file['videoID'].apply(lambda x: f'./video/{x}.mp4')
                data_file['subtitle_path'] = data_file['videoID'].apply(lambda x: f'./subtitle/{x}.srt')
                data_file['candidates'] = data_file['options'].apply(lambda x: x.tolist())

                data_file = data_file[['index', 'video', 'video_path', 'duration', 'domain', 'candidates',
                                       'sub_category', 'task_type', 'subtitle_path', 'question', 'answer']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):

        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        subtitles = ''
        if self.use_subtitle and os.path.exists(osp.join(self.data_root, line['subtitle_path'])):
            frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

            import pysubs2
            subs = pysubs2.load(osp.join(self.data_root, line['subtitle_path']), encoding='utf-8')
            subtitles = []

            for seleced_frame_id in indices:
                sub_text = ''
                cur_time = pysubs2.make_time(fps=video_info['fps'], frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace('\\N', ' ')
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = '\n'.join(subtitles)
        elif not video_llm:
            frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = self.FRAMES_TMPL_NOSUB if not self.use_subtitle else self.FRAMES_TMPL_SUB.format(subtitles)
        message.append(dict(type='text', value=text_prompt))
        question = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
        prompt = 'Question: {}\nAnswer: '.format(question)
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import extract_characters_regex, extract_option, get_dimension_rating, llm_judge_mcq

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    logger.warning('OPENAI API is not working properly, will use exact matching for evaluation')
                    logger.warning(DEBUG_MESSAGE)
                    model = None

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            lines = [data.iloc[i] for i in range(len(data))]
            tups = [(line, model) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    _evaluate_single,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['index'] == v['index'] and ans[k]['extracted'] == v['extracted']

            for idx in data['index']:
                ans_str = data.loc[data['index'] == idx, 'answer'].values[0]
                extracted = ans[idx]['extracted']
                data.loc[data['index'] == idx, 'score'] = int(extracted == ans_str)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating

    @classmethod
    def report_primary_metric(cls, metrics: dict | None) -> dict:
        if not isinstance(metrics, dict) or not metrics:
            return {}

        if 'overall|overall' in metrics:
            return {'Overall Score': metrics['overall|overall'] * 100}
        else:
            return super().report_primary_metric(metrics)
