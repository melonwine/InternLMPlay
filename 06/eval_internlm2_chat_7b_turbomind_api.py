from mmengine.config import read_base
from opencompass.models.turbomind_api import TurboMindAPIModel

with read_base():
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    dict(
        type=TurboMindAPIModel,
        abbr='internlm2-chat-7b-turbomind',
        path="internlm2-chat-7b",
        api_addr='http://0.0.0.0:23333',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
