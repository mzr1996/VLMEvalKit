import re
import logging

import numpy as np

from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item
from vlmeval.smp.file import load

FAIL_MSG = 'Failed to obtain answer via API.'

DURATIONS = [
    'short',
    'medium',
    'long',
]

DOMAINS = [
    'Knowledge',
    'Film & Television',
    'Sports Competition',
    'Artistic Performance',
    'Life Record',
    'Multilingual'
]

SUB_CATEGORIES = [
    'Humanity & History',
    'Literature & Art',
    'Biology & Medicine',
    'Finance & Commerce',
    'Astronomy',
    'Geography',
    'Law',
    'Life Tip',
    'Technology',
    'Animation',
    'Movie & TV Show',
    'Documentary',
    'News Report',
    'Esports',
    'Basketball',
    'Football',
    'Athletics',
    'Other Sports',
    'Stage Play',
    'Magic Show',
    'Variety Show',
    'Acrobatics',
    'Handicraft',
    'Food',
    'Fashion',
    'Daily Life',
    'Travel',
    'Pet & Animal',
    'Exercise',
    'Multilingual'
]

TASK_CATEGORIES = [
    'Temporal Perception',
    'Spatial Perception',
    'Attribute Perception',
    'Action Recognition',
    'Object Recognition',
    'OCR Problems',
    'Counting Problem',
    'Temporal Reasoning',
    'Spatial Reasoning',
    'Action Reasoning',
    'Object Reasoning',
    'Information Synopsis',
]


def get_dimension_rating(data_path):
    data = load(data_path)

    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ['overall']:
        duration_rating[duration] = {
            'overall': '',
            'domain': {k: [] for k in DOMAINS},
            'sub_category': {k: [] for k in SUB_CATEGORIES},
            'task_type': {k: [] for k in TASK_CATEGORIES}
        }

    for i in range(len(data)):

        domain = data.iloc[i]['domain']
        sub_ctg = data.iloc[i]['sub_category']
        task_ctg = data.iloc[i]['task_type']

        duration = data.iloc[i]['duration']
        duration_rating[duration]['domain'][domain].append(data.iloc[i]['score'])
        duration_rating[duration]['sub_category'][sub_ctg].append(data.iloc[i]['score'])
        duration_rating[duration]['task_type'][task_ctg].append(data.iloc[i]['score'])

        duration_rating['overall']['domain'][domain].append(data.iloc[i]['score'])
        duration_rating['overall']['sub_category'][sub_ctg].append(data.iloc[i]['score'])
        duration_rating['overall']['task_type'][task_ctg].append(data.iloc[i]['score'])

    for duration in DURATIONS + ['overall']:

        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.3f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.3f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['sub_category'][sub_res_dur] = sub_res_dur

        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

    return duration_rating


def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def extract_characters_regex_v2(s):
    """Extract answer letter from A-H for Video-MME-v2 (supports up to 8 options)."""
    s = s.strip()
    answer_prefixes = [
        'Final Answer:',
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[A-H]', s):
        return ''
    matches = re.search(r'[A-H]', s)
    if matches is None:
        return ''
    return matches[0]


LLM_JUDGE_PROMPT_TEMPLATE = """Given the question, the correct answer, and the model's response, determine which option ({valid_letters}) the model's response most closely matches. If the response does not match any option, respond with X.

Question: {question}
Options:
{options_block}

Model response: {model_response}

Which option does the model's response match? Respond with ONLY the letter ({valid_letters}, or X):"""


def _parse_options_list(options):
    """Parse options into a clean list of strings.

    Args:
        options: list of option strings, or a string representation of a list.

    Returns:
        list of cleaned option text strings.
    """
    if isinstance(options, str):
        try:
            import json
            options = json.loads(options)
        except (json.JSONDecodeError, TypeError):
            try:
                options = eval(options)
            except Exception:
                options = [options]

    if not isinstance(options, list):
        options = [options]

    cleaned = []
    for i, opt in enumerate(options):
        text = str(opt).strip()
        letter = chr(ord('A') + i)
        # Remove existing prefix like "A." or "A)" if present
        if text and len(text) > 1 and text[0] == letter and text[1] in '.):':
            text = text[2:].strip()
        cleaned.append(text)
    return cleaned


def _extract_option_letter(s, valid_letters):
    """Extract a valid option letter or X from LLM judge response.

    Args:
        s: The response string from the LLM judge.
        valid_letters: String of valid option letters, e.g. "ABCD".

    Returns:
        str: The matched letter or 'X'.
    """
    if not s:
        return 'X'
    s = s.strip().upper()
    # Try to find a valid option letter or X
    pattern = '[' + valid_letters + 'X]'
    match = re.search(pattern, s)
    return match.group(0) if match else 'X'


def _build_judge_prompt(question, options_list, model_response):
    """Build the LLM judge prompt dynamically based on number of options.

    Args:
        question: The question text.
        options_list: List of cleaned option text strings.
        model_response: The model's predicted response text.

    Returns:
        str: The formatted prompt for the LLM judge.
    """
    n = len(options_list)
    letters = [chr(ord('A') + i) for i in range(n)]
    valid_letters = ', '.join(letters)

    options_block = '\n'.join(
        f'({letter}) {text}' for letter, text in zip(letters, options_list)
    )

    return LLM_JUDGE_PROMPT_TEMPLATE.format(
        valid_letters=valid_letters,
        question=question,
        options_block=options_block,
        model_response=model_response,
    )


def llm_judge_mcq(model, question, options, model_response, max_options=4):
    """Use LLM as judge to match a model's open-ended response to MCQ options.

    This follows the lmms-eval approach: ask the LLM judge which option the
    model's response most closely matches, returning 'X' if no match.

    Args:
        model: An LLM client built via build_judge(), or None.
        question: The question text (without options).
        options: List of option strings, or a JSON/string representation.
        model_response: The model's predicted response text.
        max_options: Maximum number of options (default 4 for A-D). Unused if
            options list length determines the count.

    Returns:
        str: The matched option letter (A, B, C, D, ...) or 'X' if no match.
    """
    logger = logging.getLogger('Evaluation')

    parsed_options = _parse_options_list(options)
    n_options = len(parsed_options)
    valid_letters = ''.join(chr(ord('A') + i) for i in range(n_options))

    if model is None:
        # No LLM judge available; fall back to regex extraction
        regex_result = extract_characters_regex(model_response)
        return regex_result if regex_result else 'X'

    prompt = _build_judge_prompt(question, parsed_options, model_response)

    retry = 3
    while retry > 0:
        try:
            ans = model.generate(prompt)
            if 'Failed to obtain answer via API' in ans:
                logger.warning('LLM Judge API call failed, retrying...')
            else:
                result = _extract_option_letter(ans, valid_letters)
                if result in valid_letters or result == 'X':
                    return result
        except Exception as e:
            logger.warning(f'LLM Judge error: {e}')
        retry -= 1

    # All retries failed, fall back to regex
    logger.warning('LLM Judge failed after retries, falling back to regex extraction')
    regex_result = extract_characters_regex(model_response)
    return regex_result if regex_result else 'X'
